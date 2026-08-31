// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming XML tool-call parser for Qwen3-Coder.
//!
//! Qwen3-Coder emits tool calls as
//!   `<tool_call> <function=NAME> <parameter=KEY>value</parameter> ... </function> </tool_call>`
//! plus a bare `<function=...></function>` back-off form when the outer wrapper
//! is absent (shared with nemotron_nano).
//!
//! The streaming concern (buffering, chunk-split marker safety, normal_text
//! suppression) is owned by the shared [`scan::WrappedBlockScanner`]. The
//! per-block value typing is delegated to the vendored batch XML parser via
//! `parse_tool_call_block`, so a streamed call matches exactly what the batch
//! parser produces. Arguments are re-serialized in the
//! source parameter order because the v1 parser builds them from a `HashMap`
//! whose key order is non-deterministic; streaming fixtures store the arguments
//! as an exact JSON string, so order has to be pinned to the model-emitted
//! order (the order vLLM's Rust parser also preserves).

use crate::tool_calling::scan::{
    BareRecoveryLatch, InvokeEmitter, InvokeLatch, WrappedBlockScanner, WrappedBlockSpec,
    marker_prefix_suffix_len, reorder_arguments,
};
use crate::tool_calling::v1core::{ToolDefinition, XmlParserConfig, parse_tool_call_block};

use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};

const BLOCK_START: &str = "<tool_call>";
const BLOCK_END: &str = "</tool_call>";
const FUNCTION_START: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAMETER_START: &str = "<parameter=";

fn spec() -> WrappedBlockSpec {
    WrappedBlockSpec {
        family: "qwen3_coder",
        block_starts: vec![BLOCK_START.to_string()],
        block_ends: vec![BLOCK_END.to_string()],
        invoke_start: FUNCTION_START.to_string(),
        invoke_end: FUNCTION_END.to_string(),
        orphan_markers: vec![BLOCK_END.to_string()],
        // BLOCK_END is held back too so a split stray/orphan close (consumed
        // and dropped by the orphan-close handler once complete) never emits
        // its first half as text.
        holdback_markers: vec![
            BLOCK_START.to_string(),
            BLOCK_END.to_string(),
            FUNCTION_START.to_string(),
        ],
        bare_recovery_latch: BareRecoveryLatch::Set,
        invoke_latch: InvokeLatch::IfEmitted,
        drop_invoke_crossing_block_end: false,
        // Every wrapped family's markers are special tokens today.
        preserve_special_tokens: true,
        ..Default::default()
    }
}

/// Value-typing hook: types one complete `<function=...></function>` block and
/// re-orders the arguments to source order.
pub(crate) struct Qwen3Emitter {
    config: XmlParserConfig,
    tools: Vec<ToolDefinition>,
    partial: Option<PartialStringArgument>,
}

/// The only Qwen XML prefix that can safely stream before the parameter closes:
/// the first schema-guaranteed string property. `emitted_json` is exactly what
/// the final batch serializer must begin with, so the terminal delta can append
/// only its suffix without duplicating argument bytes.
struct PartialStringArgument {
    tool_index: usize,
    value_start: usize,
    emitted_raw: usize,
    emitted_json: String,
    header_emitted: bool,
    stopped: bool,
}

impl InvokeEmitter for Qwen3Emitter {
    fn parse_partial_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        if self.partial.is_none() {
            let Some((parameter, value_start)) =
                first_streamable_string_parameter(invoke, &self.tools)
            else {
                return Ok(None);
            };
            let emitted_json = format!("{{{}:\"", serde_json::to_string(&parameter)?);
            self.partial = Some(PartialStringArgument {
                tool_index,
                value_start,
                emitted_raw: 0,
                emitted_json: emitted_json.clone(),
                stopped: false,
                header_emitted: false,
            });
            return self.parse_partial_invoke(invoke, tool_index);
        }

        let partial = self.partial.as_mut().expect("initialized above");
        if partial.tool_index != tool_index || partial.stopped {
            return Ok(None);
        }
        let Some(value) = invoke.get(partial.value_start + partial.emitted_raw..) else {
            return Ok(None);
        };
        let (value, parameter_closed) = ["</parameter>", FUNCTION_END]
            .into_iter()
            .filter_map(|marker| value.find(marker).map(|end| (end, marker)))
            .min_by_key(|(end, _)| *end)
            .map(|(end, _)| (&value[..end], true))
            .unwrap_or((value, false));
        let safe_end = value.len() - qwen_partial_suffix_len(value);
        let safe = &value[..safe_end];
        let entity = safe.find('&').unwrap_or(safe.len());
        let safe = &safe[..entity];
        if entity < safe_end && safe.is_empty() {
            partial.stopped = true;
            return Ok(None);
        }
        let first_non_whitespace = safe
            .char_indices()
            .find(|(_, character)| !character.is_whitespace())
            .map(|(index, _)| index);
        let Some(first_non_whitespace) = first_non_whitespace else {
            if entity < safe_end {
                partial.stopped = true;
            }
            return Ok(None);
        };
        // The batch parser trims XML values. Discard only leading whitespace;
        // trailing whitespace stays buffered until a later non-whitespace byte
        // proves it belongs inside the final string.
        let leading = if partial.emitted_raw == 0 {
            first_non_whitespace
        } else {
            0
        };
        let safe = &safe[first_non_whitespace..];
        let end = safe
            .char_indices()
            .rev()
            .find(|(_, character)| !character.is_whitespace())
            .map(|(index, character)| index + character.len_utf8())
            .unwrap_or(0);
        if end == 0 {
            return Ok(None);
        }
        let end = value
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(value.len()))
            .take_while(|index| *index <= leading + end)
            .last()
            .expect("empty value handled above");
        let raw = &value[leading..end];
        if !partial.header_emitted {
            partial.header_emitted = true;
            return Ok(Some(ToolCallDelta {
                tool_index,
                name: Some(
                    invoke[FUNCTION_START.len()..invoke.find('>').expect("header checked")]
                        .trim()
                        .trim_matches('"')
                        .to_string(),
                ),
                arguments: partial.emitted_json.clone(),
                complete: false,
            }));
        }
        let escaped = json_string_fragment(raw);
        partial.emitted_raw += end;
        let arguments = escaped.clone();
        partial.emitted_json.push_str(&escaped);
        if entity < safe_end || parameter_closed {
            partial.stopped = true;
        }
        Ok(Some(ToolCallDelta {
            tool_index,
            name: None,
            arguments,
            complete: false,
        }))
    }

    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        // Type this ONE invoke directly. Wrapping it back in `<tool_call>` and
        // re-entering `try_tool_call_parse_xml` made the batch parser re-run
        // block discovery, which cuts the block at the FIRST `</tool_call>` —
        // so a parameter value that legitimately contains that marker was
        // truncated (`<parameter=cmd>git log </tool_call> --oneline</parameter>`
        // typed as `git log </tool_call>`). The scanner has already delimited
        // the invoke, so re-discovering its bounds could only corrupt them.
        let calls = parse_tool_call_block(invoke, &self.config, Some(&self.tools))?;
        let Some(call) = calls.into_iter().next() else {
            return Ok(None);
        };
        let arguments =
            reorder_arguments(&call.function.arguments, &source_parameter_order(invoke));
        let partial = self.partial.take();
        let streamed = partial
            .as_ref()
            .is_some_and(|partial| partial.header_emitted);
        let arguments = if let Some(partial) = &partial {
            if streamed
                && partial.tool_index == tool_index
                && arguments.starts_with(&partial.emitted_json)
            {
                arguments[partial.emitted_json.len()..].to_string()
            } else {
                arguments
            }
        } else {
            arguments
        };
        Ok(Some(ToolCallDelta {
            tool_index,
            // The streaming opener already supplied the name. The close carries
            // only the argument suffix, matching OpenAI delta semantics.
            name: (!streamed).then_some(call.function.name),
            arguments,
            complete: true,
        }))
    }

    fn reset(&mut self) {
        self.partial = None;
    }
}

/// Build the scan core for the Qwen3 tool grammar.
///
/// The single construction site for this grammar. The unified Qwen3 parser
/// calls it too and layers reasoning on top, so both parsers get the same
/// block/invoke markers, holdback set, recovery latches and value typing —
/// there is no second copy to drift.
pub(crate) fn qwen3_scanner(tools: &[Tool]) -> WrappedBlockScanner<Qwen3Emitter> {
    WrappedBlockScanner::new(
        spec(),
        Qwen3Emitter {
            config: XmlParserConfig::default(),
            tools: tools.iter().map(ToolDefinition::from).collect(),
            partial: None,
        },
    )
}

/// Return the first parameter only when its schema fixes it to a string. There
/// can be no preceding typed values in the JSON object, so its prefix is safe.
fn first_streamable_string_parameter(
    invoke: &str,
    tools: &[ToolDefinition],
) -> Option<(String, usize)> {
    let header_end = invoke.find('>')?;
    let name = invoke[FUNCTION_START.len()..header_end]
        .trim()
        .trim_matches('"')
        .trim();
    let tool = tools.iter().find(|tool| tool.name == name)?;
    let parameter_start = header_end + 1 + invoke[header_end + 1..].find(PARAMETER_START)?;
    let parameter_header = parameter_start + PARAMETER_START.len();
    let parameter_end = parameter_header + invoke[parameter_header..].find('>')?;
    let parameter = invoke[parameter_header..parameter_end]
        .trim()
        .trim_matches('"')
        .trim();
    let schema = tool
        .parameters
        .as_ref()?
        .get("properties")?
        .get(parameter)?;
    (schema.get("type").and_then(serde_json::Value::as_str) == Some("string"))
        .then(|| (parameter.to_string(), parameter_end + 1))
}

fn json_string_fragment(text: &str) -> String {
    let encoded = serde_json::to_string(text).expect("serializing a string cannot fail");
    encoded[1..encoded.len() - 1].to_string()
}

/// Keep a native function-close prefix out of an open parameter value without
/// exposing it to the shared guided-decoding marker vocabulary.
fn qwen_partial_suffix_len(value: &str) -> usize {
    marker_prefix_suffix_len(value, ["</parameter>", FUNCTION_END])
}

/// Stream parser for Qwen3-Coder XML tool calls.
pub struct Qwen3CoderToolStreamParser {
    scanner: WrappedBlockScanner<Qwen3Emitter>,
}

impl Qwen3CoderToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            scanner: qwen3_scanner(tools),
        }
    }
}

impl ToolParser for Qwen3CoderToolStreamParser {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        // Delegated, not restated: the unified adapter over this same scanner must not
        // be able to answer differently for identical markup.
        self.scanner.preserve_special_tokens()
    }

    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        self.scanner.push(chunk)
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        self.scanner.finish()
    }
}

/// Parameter names in the order they appear in a function block.
fn source_parameter_order(function: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut cursor = 0;
    while let Some(rel) = function[cursor..].find(PARAMETER_START) {
        let start = cursor + rel + PARAMETER_START.len();
        let Some(header_end) = function[start..].find('>') else {
            break;
        };
        let name = function[start..start + header_end]
            .trim()
            .trim_matches('"')
            .trim();
        if !name.is_empty() {
            names.push(name.to_string());
        }
        cursor = start + header_end + 1;
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather_tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } }
            }),
            strict: None,
        }]
    }

    fn parse_chunks(tools: &[Tool], chunks: &[&str]) -> ToolParseResult {
        let mut parser = Qwen3CoderToolStreamParser::new(tools);
        let mut out = ToolParseResult::default();
        for chunk in chunks {
            out.append(parser.push(chunk).expect("push"));
        }
        out.append(parser.finish().expect("finish"));
        out
    }

    #[test]
    fn emits_complete_call_on_close() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<tool_call> <function=get_weather>",
                " <parameter=location>",
                " NYC </parameter> </function>",
                " </tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let out = out.coalesce_calls();
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].tool_index, 0);
        assert_eq!(out.calls[0].name.as_deref(), Some("get_weather"));
        // Value is schema-typed (string) and trimmed, matching the v1 batch parser.
        assert_eq!(out.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn preserves_prefix_text_before_block() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will",
                " check the weather. <tool_call>",
                " <function=get_weather>",
                " <parameter=location>NYC</parameter> </function> </tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "I will check the weather. ");
        assert_eq!(out.calls.len(), 1);
    }

    #[test]
    fn recovers_complete_bare_function() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will check that. <function=get_weather>",
                " <parameter=location>NYC</parameter>",
                " </function>",
            ],
        );
        assert_eq!(out.normal_text, "I will check that. ");
        let out = out.coalesce_calls();
        assert_eq!(out.calls.len(), 1);
        assert_eq!(out.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn preserves_trailing_text_after_block() {
        // 8.b: trailing narration after a complete block flows into normal_text.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<tool_call> <function=get_weather> <parameter=location>NYC</parameter> </function> </tool_call>",
                " Let me know if you need more.",
            ],
        );
        assert_eq!(out.normal_text, " Let me know if you need more.");
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn preserves_inter_call_and_trailing_text() {
        // 8.d: narration between two complete blocks flows into normal_text;
        // both calls are emitted with distinct indices.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will check the weather. <tool_call> <function=get_weather> <parameter=location>NYC</parameter> </function> </tool_call>",
                " Then check LA weather. <tool_call> <function=get_weather> <parameter=location>LA</parameter> </function> </tool_call>",
            ],
        );
        assert_eq!(
            out.normal_text,
            "I will check the weather.  Then check LA weather. "
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
        assert_eq!(merged.calls[1].arguments, r#"{"location":"LA"}"#);
    }

    #[test]
    fn truncated_string_function_exposes_only_an_incomplete_delta() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "<tool_call> <function=get_weather>",
                " <parameter=location> NY",
            ],
        );
        assert_eq!(out.normal_text, "");
        assert_eq!(out.calls.len(), 2);
        assert!(out.coalesce_calls().calls.is_empty());
    }

    #[test]
    fn parameter_header_without_any_value_never_assembles_a_call() {
        let input = "<tool_call><function=get_weather><parameter=location>";
        for split in (0..=input.len()).filter(|&index| input.is_char_boundary(index)) {
            let out = parse_chunks(&weather_tools(), &[&input[..split], &input[split..]]);
            assert!(
                out.calls.is_empty(),
                "split {split} emitted without a value"
            );
            assert!(
                out.coalesce_calls().calls.is_empty(),
                "split {split} assembled an unfinished call"
            );
        }
    }

    #[test]
    fn holds_back_split_orphan_close() {
        // A stray/orphan `</tool_call>` split across a chunk boundary with no tool
        // call open must NOT leak its first half ("</tool") into normal_text: the
        // partial close is held back until the next chunk completes the marker, at
        // which point the orphan-close handler drops it entirely.
        let out = parse_chunks(&weather_tools(), &["done </tool", "_call> ok"]);
        assert!(out.calls.is_empty());
        assert!(
            !out.normal_text.contains('<'),
            "markup fragment leaked into normal_text: {:?}",
            out.normal_text
        );
        assert_eq!(out.normal_text, "done  ok");
    }

    #[test]
    fn preserves_source_parameter_order() {
        // path, old_str, new_str, command is deliberately NOT alphabetical: the
        // serialized arguments must keep the model-emitted parameter order.
        let tools = vec![Tool {
            name: "file_editor".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old_str": { "type": "string" },
                    "new_str": { "type": "string" },
                    "command": { "type": "string" }
                }
            }),
            strict: None,
        }];
        let out = parse_chunks(
            &tools,
            &[
                "<tool_call> <function=file_editor>",
                " <parameter=path>/app/x.go</parameter>",
                " <parameter=old_str>foo</parameter>",
                " <parameter=new_str>bar</parameter>",
                " <parameter=command>str_replace</parameter>",
                " </function> </tool_call>",
            ],
        );
        let out = out.coalesce_calls();
        assert_eq!(out.calls.len(), 1);
        assert_eq!(
            out.calls[0].arguments,
            r#"{"path":"/app/x.go","old_str":"foo","new_str":"bar","command":"str_replace"}"#
        );
    }

    fn stream_every_char(tools: &[Tool], input: &str) -> ToolParseResult {
        let mut parser = Qwen3CoderToolStreamParser::new(tools);
        let mut out = ToolParseResult::default();
        for character in input.chars() {
            out.append(parser.push(&character.to_string()).expect("push"));
        }
        out.append(parser.finish().expect("finish"));
        out
    }

    #[test]
    fn streams_string_arguments_before_function_close_at_every_char_boundary() {
        let input = "<tool_call><function=get_weather><parameter=location>Montréal \"café\" \\ path with substantial remaining content</parameter>still-open</function></tool_call>";
        let baseline = parse_chunks(&weather_tools(), &[input]).coalesce_calls();
        assert_eq!(
            baseline.calls[0].arguments,
            r#"{"location":"Montréal \"café\" \\ path with substantial remaining content"}"#
        );

        for split in (0..=input.len()).filter(|&index| input.is_char_boundary(index)) {
            let mut parser = Qwen3CoderToolStreamParser::new(&weather_tools());
            let mut out = ToolParseResult::default();
            out.append(parser.push(&input[..split]).expect("first push"));
            out.append(parser.push(&input[split..]).expect("second push"));
            out.append(parser.finish().expect("finish"));
            assert_eq!(
                out.coalesce_calls(),
                baseline,
                "split at byte {split} changed the completed call"
            );
        }

        let close = input.find("</function>").unwrap();
        let mut parser = Qwen3CoderToolStreamParser::new(&weather_tools());
        let mut before_close = ToolParseResult::default();
        for character in input[..close].chars() {
            before_close.append(parser.push(&character.to_string()).expect("push"));
        }
        assert!(
            before_close.calls.iter().any(|call| call.name.is_some()),
            "name-bearing delta must arrive before </function>: {before_close:?}"
        );
        let before_parameter_close = input.find("</parameter>").unwrap();
        let mut parser = Qwen3CoderToolStreamParser::new(&weather_tools());
        let mut during_value = ToolParseResult::default();
        for character in input[..before_parameter_close - 20].chars() {
            during_value.append(parser.push(&character.to_string()).expect("push"));
        }
        let emitted_arguments: String = during_value
            .calls
            .iter()
            .map(|call| call.arguments.as_str())
            .collect();
        assert!(
            emitted_arguments.contains("café"),
            "content must stream with substantial value remaining: {during_value:?}"
        );
        let out = stream_every_char(&weather_tools(), input);
        assert_eq!(
            out.calls.iter().filter(|call| call.name.is_some()).count(),
            1,
            "Qwen must put the name on the first update only"
        );
        assert!(out.calls.last().expect("call updates").complete);
        let fragments: Vec<_> = out
            .calls
            .iter()
            .filter(|call| !call.arguments.is_empty())
            .collect();
        assert!(
            fragments.len() >= 2,
            "expected multiple argument fragments, got {fragments:?}"
        );
        assert!(
            fragments
                .iter()
                .all(|fragment| !fragment.arguments.contains("</parameter>")),
            "parameter marker leaked into arguments: {fragments:?}"
        );
        assert_eq!(out.coalesce_calls(), baseline);
    }

    #[test]
    fn defers_non_string_parameter_until_function_close() {
        let tools = vec![Tool {
            name: "set_count".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "count": { "type": "integer" } }
            }),
            strict: None,
        }];
        let mut parser = Qwen3CoderToolStreamParser::new(&tools);
        let open = "<tool_call><function=set_count><parameter=count>42</parameter>";
        assert!(parser.push(open).unwrap().calls.is_empty());
        let closed = parser.push("</function></tool_call>").unwrap();
        assert_eq!(closed.calls.len(), 1);
        assert_eq!(closed.calls[0].arguments, r#"{"count":42}"#);
    }

    #[test]
    fn defers_html_entities_until_complete_typing() {
        let input = "<tool_call><function=get_weather><parameter=location>Paris &amp; Lyon</parameter></function></tool_call>";
        let entity = input.find('&').unwrap();
        let mut parser = Qwen3CoderToolStreamParser::new(&weather_tools());
        let mut before_entity = ToolParseResult::default();
        for character in input[..entity].chars() {
            before_entity.append(parser.push(&character.to_string()).expect("push"));
        }
        let emitted: String = before_entity
            .calls
            .iter()
            .map(|call| call.arguments.as_str())
            .collect();
        assert!(emitted.contains("Paris"));

        before_entity.append(parser.push(&input[entity..]).expect("push"));
        before_entity.append(parser.finish().expect("finish"));
        assert_eq!(
            before_entity
                .coalesce_calls()
                .calls
                .into_iter()
                .map(|call| call.arguments)
                .collect::<String>(),
            r#"{"location":"Paris & Lyon"}"#
        );
    }
}
