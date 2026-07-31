// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Vendored Gemma-4 call extraction: boundary resolution plus per-call typing.
//!
//! Trimmed to what v2 calls, per the module rule in `../mod.rs`. v2's streaming
//! scanner owns the whole scan, so it needs the end of ONE leading call and the
//! typing for it — not v1's whole-message span extractor, which re-derives
//! boundaries the scanner has already resolved.

mod parser;

pub use parser::{
    detect_tool_call_start_gemma4, find_leading_tool_call_end_gemma4, is_call_prefix_boundary,
    parse_one_tool_call_gemma4,
};
