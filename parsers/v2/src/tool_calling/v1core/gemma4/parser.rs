// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Reference implementation:
// https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/gemma4_tool_parser.py
//
// Gemma 4 tool-call grammar (custom, non-JSON):
//
//     <|tool_call>call:func_name{key:<|"|>value<|"|>,num:42}<tool_call|>
//
// `<|"|>`-delimited strings, bare unquoted keys, nested objects/arrays,
// multiple calls concatenated without separators.

use serde_json::{Map, Value};
use uuid::Uuid;

use super::super::ToolDefinition;
use super::super::response::{CalledFunction, ToolCallResponse, ToolCallType};

pub(crate) const TOOL_CALL_START: &str = "<|tool_call>";
pub(crate) const TOOL_CALL_END: &str = "<tool_call|>";
pub(crate) const STRING_DELIM: &str = "<|\"|>";
pub(crate) const CALL_PREFIX: &str = "call:";

fn parse_gemma_call_parts(
    name: &str,
    args_raw: &str,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<ToolCallResponse> {
    let name = name.to_string();
    if let Some(tools) = tools
        && !tools.iter().any(|t| t.name == name)
    {
        tracing::warn!(
            "Tool '{}' is not defined in the tools list (Gemma 4 parser).",
            name
        );
    }

    let args_value = match parse_args_object(args_raw) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                "Failed to parse Gemma 4 args for '{}': {}. Falling back to empty object.",
                name,
                e
            );
            Value::Object(Map::new())
        }
    };
    let arguments = serde_json::to_string(&args_value)?;

    Ok(ToolCallResponse {
        id: format!("call-{}", Uuid::new_v4()),
        tp: ToolCallType::Function,
        function: CalledFunction { name, arguments },
    })
}

fn find_balanced_args_end(input: &str, open_brace: usize) -> Option<usize> {
    debug_assert_eq!(input.as_bytes().get(open_brace), Some(&b'{'));
    let mut cursor = open_brace;
    let mut depth = 0usize;
    let mut in_string = false;

    while cursor < input.len() {
        let rest = &input[cursor..];
        if rest.starts_with(STRING_DELIM) {
            in_string = !in_string;
            cursor += STRING_DELIM.len();
            continue;
        }

        let ch = rest.chars().next()?;
        if !in_string {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth = depth.checked_sub(1)?;
                    if depth == 0 {
                        return Some(cursor);
                    }
                }
                _ => {}
            }
        }
        cursor += ch.len_utf8();
    }

    None
}

fn parse_recoverable_call_at(
    input: &str,
    allow_missing_start: bool,
    allow_missing_end: bool,
) -> Option<(&str, &str, usize)> {
    let after_start_offset = if let Some(rest) = input.strip_prefix(TOOL_CALL_START) {
        input.len() - rest.len()
    } else if allow_missing_start && input.starts_with(CALL_PREFIX) {
        0
    } else {
        return None;
    };

    let after_start = &input[after_start_offset..];
    let after_prefix = after_start.strip_prefix(CALL_PREFIX)?;
    let name_len = after_prefix.find('{').filter(|idx| *idx > 0)?;
    let name = &after_prefix[..name_len];
    if !name
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
    {
        return None;
    }

    let open_brace = after_start_offset + CALL_PREFIX.len() + name_len;
    let close_brace = find_balanced_args_end(input, open_brace)?;
    let args_start = open_brace + 1;
    let args_raw = &input[args_start..close_brace];
    let after_args = &input[close_brace + 1..];

    if after_args.starts_with(TOOL_CALL_END) {
        return Some((name, args_raw, close_brace + 1 + TOOL_CALL_END.len()));
    }

    if allow_missing_end && after_args.trim().is_empty() {
        return Some((name, args_raw, close_brace + 1));
    }

    None
}

/// True when `idx` sits on a `call:` word boundary — the preceding character is
/// not part of an identifier, so `call:` at the start of a word counts but the
/// one inside `recall:` does not.
///
/// Public because the streaming scanner asks the same question about the same
/// grammar; a second copy of this rule is a copy that drifts.
pub fn is_call_prefix_boundary(input: &str, idx: usize) -> bool {
    idx == 0
        || input[..idx]
            .chars()
            .next_back()
            .is_none_or(|ch| !(ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.')))
}

fn find_call_prefix_at_boundary(input: &str, from: usize) -> Option<usize> {
    let mut cursor = from;
    while cursor < input.len() {
        let rel = input[cursor..].find(CALL_PREFIX)?;
        let idx = cursor + rel;
        if is_call_prefix_boundary(input, idx) {
            return Some(idx);
        }
        cursor = idx + CALL_PREFIX.len();
    }
    None
}

/// Detect whether `chunk` contains the start of a Gemma 4 tool call, including
/// partial-prefix matches at the chunk boundary so streaming pipelines can hold
/// off emitting bytes that may belong to a tool-call marker.
pub fn detect_tool_call_start_gemma4(chunk: &str) -> bool {
    if chunk.contains(TOOL_CALL_START) {
        return true;
    }

    let mut cursor = 0usize;
    while let Some(idx) = find_call_prefix_at_boundary(chunk, cursor) {
        let candidate = &chunk[idx..];
        if parse_recoverable_call_at(candidate, true, true).is_some()
            || has_bare_call_body_start(candidate)
        {
            return true;
        }
        cursor = idx + CALL_PREFIX.len();
    }

    for i in 1..TOOL_CALL_START.len() {
        if TOOL_CALL_START.is_char_boundary(i) && chunk.ends_with(&TOOL_CALL_START[..i]) {
            return true;
        }
    }

    false
}

fn has_bare_call_body_start(input: &str) -> bool {
    let Some(after_prefix) = input.strip_prefix(CALL_PREFIX) else {
        return false;
    };
    let Some(open_brace) = after_prefix.find('{') else {
        return false;
    };
    if open_brace == 0 {
        return false;
    }
    after_prefix[..open_brace]
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
}

/// Returns the position immediately after the end of the ONE call that begins at
/// byte 0 of `chunk` (`call:NAME{…}<tool_call|>`, with or without a leading
/// `<|tool_call>`), or `None` while that call is still incomplete.
///
/// v1's `find_tool_call_end_position_gemma4` answers "where does the LAST
/// complete call in this blob end", which a streaming scanner cannot use
/// directly: it has to consume calls ONE at a time to keep per-call source
/// order. Both resolve the end with the same string-aware balanced scan, so a
/// `}<tool_call|>` occurring inside a `<|"|>` string value is data (`I7`).
///
/// `allow_missing_end` is end-of-stream: a body that balanced but whose
/// `<tool_call|>` never streamed is still recoverable (case `5.b`), and the whole
/// remaining text belongs to it. Mid-stream it must stay `None` so the scanner
/// keeps accumulating instead of cutting the call at the first plausible spot.
pub fn find_leading_tool_call_end_gemma4(chunk: &str, allow_missing_end: bool) -> Option<usize> {
    let starts_wrapped = chunk.starts_with(TOOL_CALL_START);
    if !starts_wrapped && !chunk.starts_with(CALL_PREFIX) {
        return None;
    }
    let allow_missing_start = !starts_wrapped;

    if let Some((_, _, consumed)) = parse_recoverable_call_at(chunk, allow_missing_start, false) {
        // Absorb a repeated close (`}<tool_call|><tool_call|>`) so the duplicate
        // does not survive as an orphan marker, matching the batch scan above.
        let mut end = consumed;
        while chunk[end..].starts_with(TOOL_CALL_END) {
            end += TOOL_CALL_END.len();
        }
        return Some(end);
    }

    if !allow_missing_end {
        return None;
    }
    // End of stream with a balanced body and no close marker. Everything left is
    // this call's markup — `parse_recoverable_call_at` only accepts the missing
    // end when what follows the body is whitespace — so the scanner consumes it
    // rather than emitting the tail as text.
    parse_recoverable_call_at(chunk, allow_missing_start, true).map(|_| chunk.len())
}

/// Parse ONE already-delimited Gemma 4 call — `[<|tool_call>]call:NAME{…}` with
/// or without its trailing `<tool_call|>` — into a typed call, or `None` if the
/// text is not a call at all.
///
/// The caller has ALREADY resolved this call's bounds (with
/// [`find_leading_tool_call_end_gemma4`]), so this must NOT re-discover them:
/// `try_tool_call_parse_gemma4` scans spans and refuses a call that is missing
/// both its opener and its closer, which is precisely the shape a streaming
/// scanner produces when it consumed the opener as block markup and the closer
/// never arrived (case `5.b`). Re-deriving bounds a caller already established is
/// also how an argument value gets truncated at a marker-looking substring
/// (`I7`), which is why the qwen3 emitter parses its block directly too.
pub fn parse_one_tool_call_gemma4(
    invoke: &str,
    tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<Option<ToolCallResponse>> {
    let allow_missing_start = !invoke.starts_with(TOOL_CALL_START);
    let Some((name, args_raw, _)) = parse_recoverable_call_at(invoke, allow_missing_start, true)
    else {
        return Ok(None);
    };
    parse_gemma_call_parts(name, args_raw, tools).map(Some)
}

// ---------------------------------------------------------------------------
// Recursive-descent parser for the Gemma 4 argument grammar
// ---------------------------------------------------------------------------
//
// Grammar (informal):
//
//   args     = (entry ("," entry)*)?
//   entry    = key ":" value
//   key      = bare-identifier (no quoting in Gemma 4 emit)
//   value    = string | number | bool | null | object | array
//   string   = "<|\"|>" .* "<|\"|>"
//   number   = -? [0-9]+ ( "." [0-9]+ )?
//   bool     = "true" | "false"
//   null     = "null" | "none" | "nil"
//   object   = "{" args "}"
//   array    = "[" (value ("," value)*)? "]"
//
// We parse straight into `serde_json::Value` so the rest of the pipeline sees
// the same shape every other parser produces.

struct Cursor<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn rest(&self) -> &'a str {
        &self.src[self.pos..]
    }

    fn eof(&self) -> bool {
        self.pos >= self.src.len()
    }

    fn skip_whitespace(&mut self) {
        let bytes = self.src.as_bytes();
        while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek_byte(&self) -> Option<u8> {
        self.src.as_bytes().get(self.pos).copied()
    }

    fn consume_byte(&mut self, b: u8) -> bool {
        if self.peek_byte() == Some(b) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
}

pub(crate) fn parse_args_object(input: &str) -> anyhow::Result<Value> {
    let mut cur = Cursor::new(input);
    cur.skip_whitespace();
    let val = parse_object_body(&mut cur)?;
    cur.skip_whitespace();
    if !cur.eof() {
        anyhow::bail!(
            "trailing characters after Gemma 4 args object at offset {}: {:?}",
            cur.pos,
            cur.rest()
        );
    }
    Ok(val)
}

fn parse_object_body(cur: &mut Cursor) -> anyhow::Result<Value> {
    let mut map = Map::new();
    cur.skip_whitespace();
    if cur.eof() || cur.peek_byte() == Some(b'}') {
        return Ok(Value::Object(map));
    }
    loop {
        cur.skip_whitespace();
        let key = parse_key(cur)?;
        cur.skip_whitespace();
        if !cur.consume_byte(b':') {
            anyhow::bail!("expected ':' after key '{}' at offset {}", key, cur.pos);
        }
        cur.skip_whitespace();
        // `key:` with no value emits `{"key": ""}` (matches upstream).
        let value = match cur.peek_byte() {
            None | Some(b',') | Some(b'}') => Value::String(String::new()),
            _ => parse_value(cur)?,
        };
        map.insert(key, value);
        cur.skip_whitespace();
        if !cur.consume_byte(b',') {
            break;
        }
    }
    Ok(Value::Object(map))
}

fn parse_key(cur: &mut Cursor) -> anyhow::Result<String> {
    let bytes = cur.src.as_bytes();
    let start = cur.pos;
    while cur.pos < bytes.len() {
        let b = bytes[cur.pos];
        if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.' {
            cur.pos += 1;
        } else {
            break;
        }
    }
    if cur.pos == start {
        anyhow::bail!("expected bare key at offset {}", start);
    }
    Ok(cur.src[start..cur.pos].to_string())
}

/// Consume `keyword` ASCII-case-insensitively, only when the next byte is not
/// a word character (so `nullable` doesn't match `null` + leftover `able`).
fn try_consume_keyword(cur: &mut Cursor, keyword: &str) -> bool {
    let bytes = cur.src.as_bytes();
    let kw = keyword.as_bytes();
    let end = cur.pos + kw.len();
    if end > bytes.len() {
        return false;
    }
    if !bytes[cur.pos..end].eq_ignore_ascii_case(kw) {
        return false;
    }
    if let Some(&next) = bytes.get(end)
        && (next.is_ascii_alphanumeric() || next == b'_')
    {
        return false;
    }
    cur.pos = end;
    true
}

fn parse_value(cur: &mut Cursor) -> anyhow::Result<Value> {
    cur.skip_whitespace();

    // Delimited string `<|"|>...<|"|>`. If the closing delimiter is missing
    // (model truncation), take everything after the opener as the value.
    if cur.rest().starts_with(STRING_DELIM) {
        cur.pos += STRING_DELIM.len();
        let body_start = cur.pos;
        match cur.src[body_start..].find(STRING_DELIM) {
            Some(end_rel) => {
                let body_end = body_start + end_rel;
                let s = cur.src[body_start..body_end].to_string();
                cur.pos = body_end + STRING_DELIM.len();
                return Ok(Value::String(s));
            }
            None => {
                let s = cur.src[body_start..].to_string();
                cur.pos = cur.src.len();
                return Ok(Value::String(s));
            }
        }
    }

    // Object
    if cur.consume_byte(b'{') {
        let v = parse_object_body(cur)?;
        cur.skip_whitespace();
        if !cur.consume_byte(b'}') {
            anyhow::bail!("expected '}}' to close object at offset {}", cur.pos);
        }
        return Ok(v);
    }

    // Array
    if cur.consume_byte(b'[') {
        return parse_array(cur);
    }

    // Booleans + null aliases (case-insensitive).
    if try_consume_keyword(cur, "true") {
        return Ok(Value::Bool(true));
    }
    if try_consume_keyword(cur, "false") {
        return Ok(Value::Bool(false));
    }
    if try_consume_keyword(cur, "null")
        || try_consume_keyword(cur, "none")
        || try_consume_keyword(cur, "nil")
    {
        return Ok(Value::Null);
    }

    // Number
    parse_number(cur)
}

fn parse_array(cur: &mut Cursor) -> anyhow::Result<Value> {
    let mut items = Vec::new();
    cur.skip_whitespace();
    if cur.consume_byte(b']') {
        return Ok(Value::Array(items));
    }
    loop {
        cur.skip_whitespace();
        items.push(parse_value(cur)?);
        cur.skip_whitespace();
        if cur.consume_byte(b']') {
            return Ok(Value::Array(items));
        }
        if !cur.consume_byte(b',') {
            anyhow::bail!("expected ',' or ']' in array at offset {}", cur.pos);
        }
    }
}

fn parse_number(cur: &mut Cursor) -> anyhow::Result<Value> {
    let start = cur.pos;
    let bytes = cur.src.as_bytes();
    if cur.peek_byte() == Some(b'-') {
        cur.pos += 1;
    }
    let int_start = cur.pos;
    while cur.pos < bytes.len() && bytes[cur.pos].is_ascii_digit() {
        cur.pos += 1;
    }
    if cur.pos == int_start {
        anyhow::bail!(
            "expected value at offset {} but got: {:?}",
            start,
            &cur.src[start..]
        );
    }
    let mut is_float = false;
    if cur.peek_byte() == Some(b'.') {
        is_float = true;
        cur.pos += 1;
        while cur.pos < bytes.len() && bytes[cur.pos].is_ascii_digit() {
            cur.pos += 1;
        }
    }
    let lex = &cur.src[start..cur.pos];
    if is_float {
        let f: f64 = lex.parse()?;
        Ok(serde_json::json!(f))
    } else {
        let i: i64 = lex.parse()?;
        Ok(serde_json::json!(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A string-typed argument whose *value* literally contains the
    /// `}<tool_call|>` end-marker sequence must NOT truncate the call there:
    /// the terminator inside a `<|"|>`-delimited string is data, and the real
    /// close is the `}<tool_call|>` that follows the closing string delimiter.
    ///
    /// This is the CodeRabbit regression: a lazy `.*?}<tool_call|>` regex stops
    /// at the first `}<tool_call|>`, even inside a string, yielding truncated
    /// arguments and a too-short removal span. The balanced, string-aware
    /// scanner ignores the terminator while `in_string`.
    #[test]
    fn end_marker_sequence_inside_string_arg_does_not_truncate() {
        let call = "<|tool_call>call:doc{note:<|\"|>use }<tool_call|> to end<|\"|>}<tool_call|>";
        let message = format!("{call} after");

        // The resolved end must cover the ENTIRE call block (start marker through
        // the REAL end marker), so the scanner strips exactly the markup and the
        // prose that follows survives as text.
        let end = find_leading_tool_call_end_gemma4(&message, false).expect("complete call");
        assert_eq!(&message[..end], call);
        assert_eq!(&message[end..], " after");

        // Arguments must be COMPLETE — the embedded `}<tool_call|>` stays inside
        // the string value rather than terminating the call early.
        let parsed = parse_one_tool_call_gemma4(&message[..end], None)
            .unwrap()
            .expect("one call");
        assert_eq!(parsed.function.name, "doc");
        assert_eq!(
            parsed.function.arguments,
            r#"{"note":"use }<tool_call|> to end"}"#
        );
    }

    /// End of stream with a balanced body and no close marker: recoverable (case
    /// `5.b`), and only at flush — mid-stream the scanner must keep accumulating
    /// rather than cut the call at the first plausible spot.
    #[test]
    fn a_body_missing_its_close_marker_resolves_only_at_flush() {
        let body = "call:get_weather{city:<|\"|>Paris<|\"|>}";
        assert_eq!(find_leading_tool_call_end_gemma4(body, false), None);
        assert_eq!(
            find_leading_tool_call_end_gemma4(body, true),
            Some(body.len())
        );
        assert_eq!(
            parse_one_tool_call_gemma4(body, None)
                .unwrap()
                .expect("recovered")
                .function
                .arguments,
            r#"{"city":"Paris"}"#
        );
    }

    /// Text that merely contains the WORD `call:` is not a call, at flush or
    /// otherwise — otherwise ordinary prose is buffered and then dropped at EOF.
    #[test]
    fn prose_containing_the_word_call_is_not_a_leading_call() {
        for flush in [false, true] {
            assert_eq!(
                find_leading_tool_call_end_gemma4("call: you tomorrow", flush),
                None
            );
        }
        assert!(
            parse_one_tool_call_gemma4("call: you tomorrow", None)
                .unwrap()
                .is_none()
        );
    }
}
