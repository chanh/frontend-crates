// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The worked example from `CUSTOM_PARSERS.md`, compiled.
//!
//! This is an INTEGRATION test on purpose: it sees exactly what a vendor crate sees
//! — the public API of `dynamo_parsers_v2` and nothing else. If a symbol stops being
//! re-exported, or the trait's required set changes, this fails to compile and the
//! documented instructions are known to be wrong before a vendor discovers it.
//!
//! Keep this file and the code blocks in `CUSTOM_PARSERS.md` in step.

use anyhow::Result;
use dynamo_parsers_v2::{Tool, UnifiedParser, UnifiedParserEvent, UnifiedParserOutput};

/// The smallest complete vendor parser: one required method plus the flush.
#[derive(Default)]
struct AcmeParser {
    /// Anything not yet safe to emit. A real parser holds a partial marker here.
    buffered: String,
}

impl UnifiedParser for AcmeParser {
    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()> {
        // A real grammar decides per byte. This one keeps a trailing '<' back,
        // standing in for "might be the start of a marker", so the example exercises
        // buffering and the flush rather than pretending neither exists.
        self.buffered.push_str(delta);
        if let Some(cut) = self.buffered.rfind('<') {
            let emit: String = self.buffered[..cut].to_string();
            self.buffered = self.buffered[cut..].to_string();
            output.push_text(emit);
        } else {
            let all = std::mem::take(&mut self.buffered);
            output.push_text(all);
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<UnifiedParserOutput> {
        let mut out = UnifiedParserOutput::default();
        // Whatever is still held back is ordinary text once the stream has ended.
        out.push_text(std::mem::take(&mut self.buffered));
        Ok(out)
    }
}

/// Removes a registration on drop, including while unwinding from a panic, so a
/// failing test cannot leave a global override installed for the next one.
struct Restore(&'static str);
impl Drop for Restore {
    fn drop(&mut self) {
        dynamo_parsers_v2::unregister_unified_parser(self.0);
    }
}

fn acme_factory(_tools: &[Tool]) -> Result<Box<dyn UnifiedParser>> {
    Ok(Box::new(AcmeParser::default()))
}

/// A vendor registers a NEW family and it is selected by name.
#[test]
fn vendor_family_is_selected_through_the_public_registry() {
    dynamo_parsers_v2::register_unified_parser("acme_doc_example", acme_factory);
    let _restore = Restore("acme_doc_example");

    let mut parser = dynamo_parsers_v2::create_unified_parser_for_family("acme_doc_example", &[])
        .expect("registered family must construct");
    let mut events = parser.push("hello ").unwrap();
    events.extend(parser.push("world").unwrap());
    events.extend(parser.finish().unwrap().events);

    let text: String = events
        .iter()
        .map(|e| match e {
            UnifiedParserEvent::Text(t) => t.as_str(),
            _ => "",
        })
        .collect();
    assert_eq!(text, "hello world");
}

/// Nothing is lost across an arbitrary split — the property the contract section of
/// `CUSTOM_PARSERS.md` asks a vendor to test, demonstrated on the example itself.
#[test]
fn example_parser_is_split_invariant() {
    let input = "alpha <not-a-marker> beta";
    let whole = {
        let mut p = AcmeParser::default();
        let mut out = UnifiedParserOutput::default();
        p.parse_into(input, &mut out).unwrap();
        let mut tail = p.finish().unwrap();
        out.append(&mut tail);
        out.assembled()
    };

    for cut in 1..input.len() {
        if !input.is_char_boundary(cut) {
            continue;
        }
        let mut p = AcmeParser::default();
        let mut out = UnifiedParserOutput::default();
        p.parse_into(&input[..cut], &mut out).unwrap();
        p.parse_into(&input[cut..], &mut out).unwrap();
        let mut tail = p.finish().unwrap();
        out.append(&mut tail);
        assert_eq!(
            out.assembled(),
            whole,
            "split at {cut} produced a different result than the whole input"
        );
    }
}

/// A vendor replaces a family this crate ships, then gives it back.
#[test]
fn vendor_can_shadow_a_builtin_family() {
    let family = "qwen3";
    assert!(dynamo_parsers_v2::builtin_unified_families().contains(&family));

    let builtin = dynamo_parsers_v2::create_unified_parser_for_family(family, &[])
        .unwrap()
        .push("<think>hi</think>")
        .unwrap();

    dynamo_parsers_v2::register_unified_parser(family, acme_factory);
    let restore = Restore(family);
    let shadowed = dynamo_parsers_v2::create_unified_parser_for_family(family, &[])
        .unwrap()
        .push("<think>hi</think>")
        .unwrap();
    assert_ne!(
        shadowed, builtin,
        "the vendor parser must be the one that ran"
    );

    drop(restore);
    let restored = dynamo_parsers_v2::create_unified_parser_for_family(family, &[])
        .unwrap()
        .push("<think>hi</think>")
        .unwrap();
    assert_eq!(restored, builtin, "unregistering must restore the built-in");
}
