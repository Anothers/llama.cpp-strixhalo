#pragma once

#include "server-common.h"

#include <cstddef>
#include <string>
#include <vector>

// Normalization of Anthropic Messages content blocks into OpenAI chat content parts.
//
// Shared by /v1/messages and /v1/messages/count_tokens (both go through
// server_chat_convert_anthropic_to_oai), so the two always agree.
//
//  - text      -> text part (kept as-is)
//  - image     -> image_url part with a data: URI (base64 source only)
//  - document  -> text part(s) framed with the title/context; PDFs become one JPEG image per page
//  - tool_result content is normalized with the same rules (documents inside it included)
//
// Never fetches URLs or resolves Files API ids: those sources are rejected. Invalid input
// (bad base64, invalid UTF-8, corrupt/encrypted PDF, limits exceeded) throws std::invalid_argument,
// which the HTTP layer turns into a 400. Raw base64 is never copied into prompt text.

struct anthropic_content_limits {
    size_t max_decoded_bytes  = 20u << 20;  // per document, after base64 decoding
    int    max_pdf_pages      = 32;         // per PDF document
    int    max_page_side_px   = 2048;       // longest side of a rendered page
    size_t max_rendered_bytes = 64u << 20;  // encoded page images, whole request

    // defaults, overridable with LLAMA_ARG_ANTHROPIC_DOC_{MAX_BYTES,MAX_PAGES,MAX_SIDE,MAX_RENDERED}
    static anthropic_content_limits from_env();
};

struct anthropic_content_ctx {
    anthropic_content_limits limits;
    size_t rendered_bytes = 0;  // running total for max_rendered_bytes
    int    document_index = 0;  // 1-based numbering of documents in the request
};

// Convert one block array (user message content or tool_result content) into OpenAI content parts,
// preserving block order. Unknown block types are skipped, as before.
json anthropic_blocks_to_oai_parts(const json & blocks, anthropic_content_ctx & ctx);

// true when the parts contain anything other than text (the caller keeps array form then)
bool anthropic_parts_have_media(const json & parts);

// Render a PDF into JPEG page images (implemented in server-pdf.cpp when built with poppler).
// Throws std::invalid_argument for corrupt, encrypted or over-limit documents.
bool anthropic_pdf_supported();
std::vector<std::string> anthropic_render_pdf(const std::string & pdf_bytes, anthropic_content_ctx & ctx);
