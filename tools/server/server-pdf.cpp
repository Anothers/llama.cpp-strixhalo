// PDF -> JPEG page images for Anthropic document blocks (built only when poppler-cpp and libjpeg are found).
#include "server-anthropic-content.h"

#include <poppler-document.h>
#include <poppler-global.h>
#include <poppler-image.h>
#include <poppler-page.h>
#include <poppler-page-renderer.h>

#include <jpeglib.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>

static void pdf_silence(const std::string &, void *) {}

bool anthropic_pdf_supported() { return true; }

static std::string encode_jpeg(const unsigned char * rgb, int w, int h, int quality) {
    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    unsigned char * buf = nullptr;
    unsigned long size = 0;
    jpeg_mem_dest(&cinfo, &buf, &size);
    cinfo.image_width = w;
    cinfo.image_height = h;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);
    while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPROW row = const_cast<unsigned char *>(rgb + (size_t) cinfo.next_scanline * w * 3);
        jpeg_write_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_compress(&cinfo);
    std::string out((const char *) buf, size);
    jpeg_destroy_compress(&cinfo);
    free(buf);
    return out;
}

std::vector<std::string> anthropic_render_pdf(const std::string & pdf_bytes, anthropic_content_ctx & ctx) {
    static const bool quiet = [] { poppler::set_debug_error_function(pdf_silence, nullptr); return true; }();
    (void) quiet;

    const auto & lim = ctx.limits;
    if (pdf_bytes.size() < 5 || pdf_bytes.compare(0, 5, "%PDF-") != 0) {
        throw std::invalid_argument("PDF document: data is not a PDF (missing %PDF- header)");
    }
    poppler::byte_array raw(pdf_bytes.begin(), pdf_bytes.end());
    std::unique_ptr<poppler::document> doc(poppler::document::load_from_data(&raw));
    if (!doc) {
        throw std::invalid_argument("PDF document: the file is corrupt or cannot be parsed");
    }
    if (doc->is_locked() || doc->is_encrypted()) {
        throw std::invalid_argument("PDF document: encrypted PDFs are not supported");
    }
    const int n = doc->pages();
    if (n <= 0) {
        throw std::invalid_argument("PDF document: the file has no pages");
    }
    if (n > lim.max_pdf_pages) {
        throw std::invalid_argument("PDF document: " + std::to_string(n) + " pages exceeds the limit of " +
                                    std::to_string(lim.max_pdf_pages) + " pages");
    }

    poppler::page_renderer renderer;
    renderer.set_render_hint(poppler::page_renderer::antialiasing, true);
    renderer.set_render_hint(poppler::page_renderer::text_antialiasing, true);
    renderer.set_image_format(poppler::image::format_argb32);

    std::vector<std::string> pages;
    pages.reserve(n);
    for (int i = 0; i < n; ++i) {
        std::unique_ptr<poppler::page> page(doc->create_page(i));
        if (!page) {
            throw std::invalid_argument("PDF document: page " + std::to_string(i + 1) + " cannot be read");
        }
        const poppler::rectf box = page->page_rect(poppler::media_box);
        double w_pt = box.width(), h_pt = box.height();
        if (page->orientation() == poppler::page::landscape || page->orientation() == poppler::page::seascape) {
            std::swap(w_pt, h_pt);
        }
        const double longest_pt = std::max(w_pt, h_pt);
        if (!(longest_pt > 0)) {
            throw std::invalid_argument("PDF document: page " + std::to_string(i + 1) + " has an invalid size");
        }
        // 144 dpi (2x) for readable text, reduced so that the longest side stays within the limit
        const double dpi = std::min(144.0, std::floor(lim.max_page_side_px * 72.0 / longest_pt));
        poppler::image img = renderer.render_page(page.get(), dpi, dpi);
        if (!img.is_valid() || img.width() <= 0 || img.height() <= 0) {
            throw std::invalid_argument("PDF document: page " + std::to_string(i + 1) + " failed to render");
        }
        const int w = std::min(img.width(),  lim.max_page_side_px);
        const int h = std::min(img.height(), lim.max_page_side_px);
        // argb32 is stored as native-endian 0xAARRGGBB: B, G, R, A in memory on little-endian hosts
        std::vector<unsigned char> rgb((size_t) w * h * 3);
        const char * src = img.const_data();
        const int stride = img.bytes_per_row();
        for (int y = 0; y < h; ++y) {
            const unsigned char * row = (const unsigned char *) src + (size_t) y * stride;
            unsigned char * dst = rgb.data() + (size_t) y * w * 3;
            for (int x = 0; x < w; ++x) {
                dst[3 * x + 0] = row[4 * x + 2];
                dst[3 * x + 1] = row[4 * x + 1];
                dst[3 * x + 2] = row[4 * x + 0];
            }
        }
        std::string jpg = encode_jpeg(rgb.data(), w, h, 85);
        ctx.rendered_bytes += jpg.size();
        if (ctx.rendered_bytes > lim.max_rendered_bytes) {
            throw std::invalid_argument("PDF document: rendered page images exceed the limit of " +
                                        std::to_string(lim.max_rendered_bytes) + " bytes");
        }
        pages.push_back(std::move(jpg));
    }
    return pages;
}
