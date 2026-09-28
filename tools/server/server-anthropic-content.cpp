#include "server-anthropic-content.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

anthropic_content_limits anthropic_content_limits::from_env() {
    anthropic_content_limits l;
    auto env_num = [](const char * name, long long def) -> long long {
        const char * v = std::getenv(name);
        if (!v || !*v) {
            return def;
        }
        char * end = nullptr;
        const long long x = std::strtoll(v, &end, 10);
        return (end && *end == '\0' && x > 0) ? x : def;
    };
    l.max_decoded_bytes  = (size_t) env_num("LLAMA_ARG_ANTHROPIC_DOC_MAX_BYTES",    (long long) l.max_decoded_bytes);
    l.max_pdf_pages      = (int)    env_num("LLAMA_ARG_ANTHROPIC_DOC_MAX_PAGES",    l.max_pdf_pages);
    l.max_page_side_px   = (int)    env_num("LLAMA_ARG_ANTHROPIC_DOC_MAX_SIDE",     l.max_page_side_px);
    l.max_rendered_bytes = (size_t) env_num("LLAMA_ARG_ANTHROPIC_DOC_MAX_RENDERED", (long long) l.max_rendered_bytes);
    return l;
}

// ---------------------------------------------------------------------------------------------------------------
// strict helpers

static int b64_value(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

// Standard base64 with '=' padding; no whitespace, no URL-safe alphabet. Throws on anything else.
static std::string strict_base64_decode(const std::string & in, size_t max_bytes, const std::string & what) {
    if (in.empty()) {
        throw std::invalid_argument(what + ": empty base64 data");
    }
    if (in.size() % 4 != 0) {
        throw std::invalid_argument(what + ": invalid base64 (length is not a multiple of 4)");
    }
    const size_t pad = (in[in.size() - 1] == '=') + (in.size() >= 2 && in[in.size() - 2] == '=');
    const size_t out_len = in.size() / 4 * 3 - pad;
    if (out_len > max_bytes) {
        throw std::invalid_argument(what + ": decoded size " + std::to_string(out_len) + " bytes exceeds the limit of " +
                                    std::to_string(max_bytes) + " bytes");
    }
    std::string out;
    out.reserve(out_len);
    for (size_t i = 0; i < in.size(); i += 4) {
        int v[4];
        for (int j = 0; j < 4; ++j) {
            const unsigned char c = (unsigned char) in[i + j];
            if (c == '=' && i + 4 == in.size() && j >= 2 && (j == 3 || in[i + 3] == '=')) {
                v[j] = 0;
                continue;
            }
            v[j] = b64_value(c);
            if (v[j] < 0) {
                throw std::invalid_argument(what + ": invalid base64 character at offset " + std::to_string(i + j));
            }
        }
        const uint32_t triple = (uint32_t(v[0]) << 18) | (uint32_t(v[1]) << 12) | (uint32_t(v[2]) << 6) | uint32_t(v[3]);
        out.push_back((char) ((triple >> 16) & 0xFF));
        if (!(i + 4 == in.size() && pad == 2)) out.push_back((char) ((triple >> 8) & 0xFF));
        if (!(i + 4 == in.size() && pad >= 1)) out.push_back((char) (triple & 0xFF));
    }
    return out;
}

static bool is_valid_utf8_strict(const std::string & s) {
    const auto * p = (const unsigned char *) s.data();
    const size_t n = s.size();
    size_t i = 0;
    while (i < n) {
        const unsigned char c = p[i];
        if (c < 0x80) { ++i; continue; }
        size_t len; uint32_t cp;
        if ((c & 0xE0) == 0xC0)      { len = 2; cp = c & 0x1F; }
        else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; }
        else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; }
        else return false;
        if (i + len > n) return false;
        for (size_t k = 1; k < len; ++k) {
            if ((p[i + k] & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (p[i + k] & 0x3F);
        }
        if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && (cp < 0x10000 || cp > 0x10FFFF)) ||
            (cp >= 0xD800 && cp <= 0xDFFF)) {
            return false;  // overlong, out of range or surrogate
        }
        i += len;
    }
    return true;
}

static std::string base64_encode_bytes(const std::string & in) {
    static const char * tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        const uint32_t t = (uint32_t((unsigned char) in[i]) << 16) | (uint32_t((unsigned char) in[i + 1]) << 8) | (unsigned char) in[i + 2];
        out.push_back(tbl[(t >> 18) & 63]); out.push_back(tbl[(t >> 12) & 63]); out.push_back(tbl[(t >> 6) & 63]); out.push_back(tbl[t & 63]);
    }
    if (i < in.size()) {
        uint32_t t = uint32_t((unsigned char) in[i]) << 16;
        if (i + 1 < in.size()) t |= uint32_t((unsigned char) in[i + 1]) << 8;
        out.push_back(tbl[(t >> 18) & 63]); out.push_back(tbl[(t >> 12) & 63]);
        out.push_back(i + 1 < in.size() ? tbl[(t >> 6) & 63] : '=');
        out.push_back('=');
    }
    return out;
}

static std::string xml_attr_escape(const std::string & s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '&': o += "&amp;";  break;
            case '"': o += "&quot;"; break;
            case '<': o += "&lt;";   break;
            case '>': o += "&gt;";   break;
            case '\n': o += ' ';     break;
            default:  o.push_back(c);
        }
    }
    return o;
}

static json text_part(const std::string & text) {
    return json{{"type", "text"}, {"text", text}};
}

static json image_part(const std::string & media_type, const std::string & b64) {
    return json{{"type", "image_url"}, {"image_url", {{"url", "data:" + media_type + ";base64," + b64}}}};
}

// ---------------------------------------------------------------------------------------------------------------
// blocks

static void reject_remote_source(const std::string & source_type, const std::string & what) {
    if (source_type == "url") {
        throw std::invalid_argument(what + ": URL sources are not supported (the server does not fetch URLs); send the data as base64");
    }
    if (source_type == "file") {
        throw std::invalid_argument(what + ": Files API file_id sources are not supported; send the data inline");
    }
}

static void image_block_to_parts(const json & block, json & out, const anthropic_content_ctx & ctx) {
    json source = json_value(block, "source", json::object());
    const std::string source_type = json_value(source, "type", std::string());
    reject_remote_source(source_type, "image block");
    if (source_type != "base64") {
        throw std::invalid_argument("image block: unsupported source type '" + source_type + "'");
    }
    const std::string media_type = json_value(source, "media_type", std::string("image/jpeg"));
    if (media_type.rfind("image/", 0) != 0) {
        throw std::invalid_argument("image block: media_type must be image/*, got '" + media_type + "'");
    }
    const std::string data = json_value(source, "data", std::string());
    strict_base64_decode(data, ctx.limits.max_decoded_bytes, "image block");  // validation only
    out.push_back(image_part(media_type, data));
}

static void document_block_to_parts(const json & block, json & out, anthropic_content_ctx & ctx);

static void blocks_to_parts(const json & blocks, json & out, anthropic_content_ctx & ctx) {
    for (const auto & block : blocks) {
        const std::string type = json_value(block, "type", std::string());
        if (type == "text") {
            out.push_back(block);
        } else if (type == "image") {
            image_block_to_parts(block, out, ctx);
        } else if (type == "document") {
            document_block_to_parts(block, out, ctx);
        }
        // other block types (thinking, tool_use, tool_result, ...) are handled by the caller or dropped as before
    }
}

static void document_block_to_parts(const json & block, json & out, anthropic_content_ctx & ctx) {
    const int index = ++ctx.document_index;
    const std::string what = "document block #" + std::to_string(index);

    json source = json_value(block, "source", json::object());
    const std::string source_type = json_value(source, "type", std::string());
    const std::string title   = json_value(block, "title", std::string());
    const std::string context = json_value(block, "context", std::string());

    reject_remote_source(source_type, what);

    auto header = [&](const std::string & extra) {
        std::string h = "<document index=\"" + std::to_string(index) + "\"";
        if (!title.empty()) {
            h += " title=\"" + xml_attr_escape(title) + "\"";
        }
        h += extra + ">\n";
        if (!context.empty()) {
            h += "<document_context>\n" + context + "\n</document_context>\n";
        }
        return h;
    };
    const std::string footer = "</document>";

    auto text_document = [&](const std::string & text) {
        if (text.size() > ctx.limits.max_decoded_bytes) {
            throw std::invalid_argument(what + ": text size " + std::to_string(text.size()) + " bytes exceeds the limit of " +
                                        std::to_string(ctx.limits.max_decoded_bytes) + " bytes");
        }
        if (!is_valid_utf8_strict(text)) {
            throw std::invalid_argument(what + ": text is not valid UTF-8");
        }
        out.push_back(text_part(header("") + "<document_content>\n" + text + "\n</document_content>\n" + footer));
    };

    if (source_type == "text") {
        const std::string media_type = json_value(source, "media_type", std::string("text/plain"));
        if (media_type.rfind("text/", 0) != 0) {
            throw std::invalid_argument(what + ": text source media_type must be text/*, got '" + media_type + "'");
        }
        text_document(json_value(source, "data", std::string()));
        return;
    }

    if (source_type == "content") {
        // custom content document: an ordered list of text/image blocks
        json content = json_value(source, "content", json());
        out.push_back(text_part(header("")));
        if (content.is_string()) {
            out.push_back(text_part(content.get<std::string>()));
        } else if (content.is_array()) {
            for (const auto & c : content) {
                const std::string t = json_value(c, "type", std::string());
                if (t != "text" && t != "image") {
                    throw std::invalid_argument(what + ": content source only accepts text and image blocks, got '" + t + "'");
                }
            }
            blocks_to_parts(content, out, ctx);
        } else {
            throw std::invalid_argument(what + ": content source needs a 'content' array or string");
        }
        out.push_back(text_part(footer));
        return;
    }

    if (source_type == "base64") {
        const std::string media_type = json_value(source, "media_type", std::string());
        const std::string bytes = strict_base64_decode(json_value(source, "data", std::string()), ctx.limits.max_decoded_bytes, what);
        if (media_type.rfind("text/", 0) == 0) {
            text_document(bytes);
            return;
        }
        if (media_type == "application/pdf") {
            if (!anthropic_pdf_supported()) {
                throw std::invalid_argument(what + ": PDF documents are not supported by this server build");
            }
            const std::vector<std::string> pages = anthropic_render_pdf(bytes, ctx);
            const int n = (int) pages.size();
            out.push_back(text_part(header(" media_type=\"application/pdf\" pages=\"" + std::to_string(n) + "\"")));
            for (int p = 0; p < n; ++p) {
                out.push_back(text_part("<page number=\"" + std::to_string(p + 1) + "\"/>"));
                out.push_back(image_part("image/jpeg", base64_encode_bytes(pages[p])));
            }
            out.push_back(text_part(footer));
            return;
        }
        throw std::invalid_argument(what + ": unsupported base64 media_type '" + media_type + "' (supported: text/*, application/pdf)");
    }

    throw std::invalid_argument(what + ": unsupported source type '" + source_type + "'");
}

json anthropic_blocks_to_oai_parts(const json & blocks, anthropic_content_ctx & ctx) {
    json out = json::array();
    blocks_to_parts(blocks, out, ctx);
    return out;
}

bool anthropic_parts_have_media(const json & parts) {
    for (const auto & p : parts) {
        if (json_value(p, "type", std::string()) != "text") {
            return true;
        }
    }
    return false;
}

#ifndef LLAMA_SERVER_PDF
bool anthropic_pdf_supported() { return false; }
std::vector<std::string> anthropic_render_pdf(const std::string &, anthropic_content_ctx &) {
    throw std::invalid_argument("PDF documents are not supported by this server build");
}
#endif
