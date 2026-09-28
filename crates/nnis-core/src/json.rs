//! Minimal strict JSON (RFC 8259) value model, parser, and writer.
//!
//! Crate-private support for versioned portable records. The parser rejects
//! duplicate object keys, trailing content, leading zeros, lone surrogates,
//! control characters in strings, excessive nesting, and oversized input.
//! Numbers keep their exact source text so integer fields are never routed
//! through floating point.

use core::fmt::Write as _;

/// Maximum accepted input size in bytes.
pub(crate) const MAX_JSON_BYTES: usize = 16 * 1024 * 1024;

/// Maximum nesting depth of arrays and objects.
pub(crate) const MAX_JSON_DEPTH: usize = 64;

/// Parsed JSON value.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    /// Exact number text, already checked against the JSON grammar.
    Number(String),
    String(String),
    Array(Vec<Json>),
    /// Members in source order with unique keys.
    Object(Vec<(String, Json)>),
}

/// JSON syntax error with the byte offset where parsing failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JsonSyntaxError {
    pub(crate) offset: usize,
    pub(crate) reason: &'static str,
}

pub(crate) fn parse(input: &str) -> Result<Json, JsonSyntaxError> {
    if input.len() > MAX_JSON_BYTES {
        return Err(JsonSyntaxError {
            offset: 0,
            reason: "input too large",
        });
    }
    let mut parser = Parser {
        bytes: input.as_bytes(),
        position: 0,
    };
    parser.whitespace();
    let value = parser.value(0)?;
    parser.whitespace();
    if parser.position != parser.bytes.len() {
        return Err(parser.error("trailing content"));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Parser<'_> {
    fn error(&self, reason: &'static str) -> JsonSyntaxError {
        JsonSyntaxError {
            offset: self.position,
            reason,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    fn whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.position += 1;
        }
    }

    fn literal(&mut self, text: &'static [u8], value: Json) -> Result<Json, JsonSyntaxError> {
        if self.bytes[self.position..].starts_with(text) {
            self.position += text.len();
            Ok(value)
        } else {
            Err(self.error("invalid literal"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, JsonSyntaxError> {
        match self.peek() {
            Some(b'n') => self.literal(b"null", Json::Null),
            Some(b't') => self.literal(b"true", Json::Bool(true)),
            Some(b'f') => self.literal(b"false", Json::Bool(false)),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b'[') => self.array(depth + 1),
            Some(b'{') => self.object(depth + 1),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(self.error("expected a value")),
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, JsonSyntaxError> {
        if depth > MAX_JSON_DEPTH {
            return Err(self.error("nesting too deep"));
        }
        self.position += 1;
        let mut items = Vec::new();
        self.whitespace();
        if self.peek() == Some(b']') {
            self.position += 1;
            return Ok(Json::Array(items));
        }
        loop {
            self.whitespace();
            items.push(self.value(depth)?);
            self.whitespace();
            match self.peek() {
                Some(b',') => self.position += 1,
                Some(b']') => {
                    self.position += 1;
                    return Ok(Json::Array(items));
                }
                _ => return Err(self.error("expected ',' or ']'")),
            }
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, JsonSyntaxError> {
        if depth > MAX_JSON_DEPTH {
            return Err(self.error("nesting too deep"));
        }
        self.position += 1;
        let mut members: Vec<(String, Json)> = Vec::new();
        self.whitespace();
        if self.peek() == Some(b'}') {
            self.position += 1;
            return Ok(Json::Object(members));
        }
        loop {
            self.whitespace();
            if self.peek() != Some(b'"') {
                return Err(self.error("expected a string key"));
            }
            let key_offset = self.position;
            let key = self.string()?;
            if members.iter().any(|(existing, _)| *existing == key) {
                return Err(JsonSyntaxError {
                    offset: key_offset,
                    reason: "duplicate object key",
                });
            }
            self.whitespace();
            if self.peek() != Some(b':') {
                return Err(self.error("expected ':'"));
            }
            self.position += 1;
            self.whitespace();
            let value = self.value(depth)?;
            members.push((key, value));
            self.whitespace();
            match self.peek() {
                Some(b',') => self.position += 1,
                Some(b'}') => {
                    self.position += 1;
                    return Ok(Json::Object(members));
                }
                _ => return Err(self.error("expected ',' or '}'")),
            }
        }
    }

    fn number(&mut self) -> Result<Json, JsonSyntaxError> {
        let start = self.position;
        if self.peek() == Some(b'-') {
            self.position += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.position += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.error("leading zero"));
                }
            }
            Some(b'1'..=b'9') => self.digits(),
            _ => return Err(self.error("expected digit")),
        }
        if self.peek() == Some(b'.') {
            self.position += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("expected fraction digit"));
            }
            self.digits();
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.position += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.position += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(self.error("expected exponent digit"));
            }
            self.digits();
        }
        let text = core::str::from_utf8(&self.bytes[start..self.position])
            .map_err(|_| self.error("invalid number"))?;
        Ok(Json::Number(text.to_owned()))
    }

    fn digits(&mut self) {
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.position += 1;
        }
    }

    fn hex4(&mut self) -> Result<u32, JsonSyntaxError> {
        let mut value = 0u32;
        for _ in 0..4 {
            let digit = match self.peek() {
                Some(byte @ b'0'..=b'9') => u32::from(byte - b'0'),
                Some(byte @ b'a'..=b'f') => u32::from(byte - b'a' + 10),
                Some(byte @ b'A'..=b'F') => u32::from(byte - b'A' + 10),
                _ => return Err(self.error("invalid unicode escape")),
            };
            value = value * 16 + digit;
            self.position += 1;
        }
        Ok(value)
    }

    fn string(&mut self) -> Result<String, JsonSyntaxError> {
        self.position += 1;
        let mut output = String::new();
        loop {
            let run_start = self.position;
            while let Some(byte) = self.peek() {
                if byte == b'"' || byte == b'\\' || byte < 0x20 {
                    break;
                }
                self.position += 1;
            }
            // Input is a &str and runs stop only at ASCII bytes, so each run
            // is valid UTF-8.
            output.push_str(
                core::str::from_utf8(&self.bytes[run_start..self.position])
                    .map_err(|_| self.error("invalid UTF-8"))?,
            );
            match self.peek() {
                Some(b'"') => {
                    self.position += 1;
                    return Ok(output);
                }
                Some(b'\\') => {
                    self.position += 1;
                    let escape = self
                        .peek()
                        .ok_or_else(|| self.error("unterminated escape"))?;
                    self.position += 1;
                    match escape {
                        b'"' => output.push('"'),
                        b'\\' => output.push('\\'),
                        b'/' => output.push('/'),
                        b'b' => output.push('\u{8}'),
                        b'f' => output.push('\u{c}'),
                        b'n' => output.push('\n'),
                        b'r' => output.push('\r'),
                        b't' => output.push('\t'),
                        b'u' => {
                            let first = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&first) {
                                if !self.bytes[self.position..].starts_with(b"\\u") {
                                    return Err(self.error("lone surrogate"));
                                }
                                self.position += 2;
                                let second = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&second) {
                                    return Err(self.error("lone surrogate"));
                                }
                                0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&first) {
                                return Err(self.error("lone surrogate"));
                            } else {
                                first
                            };
                            output.push(
                                char::from_u32(code)
                                    .ok_or_else(|| self.error("invalid code point"))?,
                            );
                        }
                        _ => return Err(self.error("invalid escape")),
                    }
                }
                Some(_) => return Err(self.error("control character in string")),
                None => return Err(self.error("unterminated string")),
            }
        }
    }
}

/// Write `value` as compact JSON with members in the given order.
pub(crate) fn write(value: &Json, output: &mut String) {
    match value {
        Json::Null => output.push_str("null"),
        Json::Bool(true) => output.push_str("true"),
        Json::Bool(false) => output.push_str("false"),
        Json::Number(text) => output.push_str(text),
        Json::String(text) => write_string(text, output),
        Json::Array(items) => {
            output.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write(item, output);
            }
            output.push(']');
        }
        Json::Object(members) => {
            output.push('{');
            for (index, (key, item)) in members.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_string(key, output);
                output.push(':');
                write(item, output);
            }
            output.push('}');
        }
    }
}

fn write_string(text: &str, output: &mut String) {
    output.push('"');
    for character in text.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            control if u32::from(control) < 0x20 => {
                let _ = write!(output, "\\u{:04x}", u32::from(control));
            }
            other => output.push(other),
        }
    }
    output.push('"');
}

/// JSON number text for an unsigned integer.
pub(crate) fn unsigned(value: u64) -> Json {
    Json::Number(value.to_string())
}

/// JSON number text for a finite float using Rust's shortest round-trip form.
///
/// Returns `None` for non-finite values, which JSON cannot represent.
pub(crate) fn float(value: f64) -> Option<Json> {
    if value.is_finite() {
        Some(Json::Number(format!("{value:?}")))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_writes_nested_values() {
        let text = r#"{"a":[1,-2.5e3,true,false,null],"b":{"c":"x\"\\\n\u00e9\ud83d\ude00"}}"#;
        let value = parse(text).unwrap();
        let mut output = String::new();
        write(&value, &mut output);
        assert_eq!(parse(&output).unwrap(), value);
        match &value {
            Json::Object(members) => {
                assert_eq!(members[0].0, "a");
                assert_eq!(
                    members[1].1,
                    Json::Object(vec![("c".into(), Json::String("x\"\\\né😀".into()))])
                );
            }
            _ => panic!("expected object"),
        }
    }

    #[test]
    fn rejects_malformed_input() {
        for bad in [
            "",
            "{",
            "[1,]",
            "{\"a\":1,}",
            "{\"a\":1,\"a\":2}",
            "01",
            "1.",
            "1e",
            "-",
            "\"\\x\"",
            "\"\\ud800\"",
            "\"\\udc00\"",
            "\"a\u{1}b\"",
            "nul",
            "1 2",
            "{a:1}",
        ] {
            assert!(parse(bad).is_err(), "{bad:?} must be rejected");
        }
        let deep = "[".repeat(MAX_JSON_DEPTH + 1) + &"]".repeat(MAX_JSON_DEPTH + 1);
        assert_eq!(parse(&deep).unwrap_err().reason, "nesting too deep");
        let ok = "[".repeat(MAX_JSON_DEPTH) + &"]".repeat(MAX_JSON_DEPTH);
        assert!(parse(&ok).is_ok());
    }

    #[test]
    fn floats_round_trip_exactly_and_non_finite_is_unrepresentable() {
        for value in [0.1, 1.0, -0.0, 1e-300, 1.7976931348623157e308, 123456.789] {
            let Some(Json::Number(text)) = float(value) else {
                panic!("finite value")
            };
            assert_eq!(text.parse::<f64>().unwrap().to_bits(), value.to_bits());
            assert!(parse(&text).is_ok());
        }
        assert_eq!(float(f64::NAN), None);
        assert_eq!(float(f64::INFINITY), None);
        assert_eq!(
            unsigned(u64::MAX),
            Json::Number("18446744073709551615".into())
        );
    }
}
