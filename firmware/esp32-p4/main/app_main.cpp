// tensor-factory on the ESP32-P4 (ESP-DL, int8).
//
// 1. Sweep: every model in models.inc, activations in PSRAM and then internal RAM first.
//    One "BENCH {json}" line each.
// 2. Image files: every JPEG in jpegs.inc is decoded, resized and detected on the board.
//    One "DET {json}" line each, scored on the host by tools/score.py.
// 3. Soak: the model in detect.inc on the frames in frames.inc, forever. Each frame goes
//    through the whole path (uint8 RGB -> int8 input -> network -> soft-argmax box) and is
//    checked two ways: against what the desktop int8 graph produced for that frame, and
//    bit-for-bit against this board's own first answer. One "SOAK {json}" line per round.
// 4. Serve (when nothing is compiled in to soak): the model in the "model" flash partition,
//    written there by tools/deploy.py with no rebuild. One "MODEL {json}" line, then
//    "READY"; each "IMG <base64 JPEG>" line on the console gets one "DET {json}" line back,
//    and "PING <token>" gets "READY <token>".
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "dl_image_jpeg.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_model_base.hpp"
#include "driver/uart.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_rom_crc.h"
#include "esp_timer.h"

struct Entry {
    const char *name;
    const uint8_t *data;
};

#define MODEL(sym) extern const uint8_t _binary_##sym##_espdl_start[];
#include "models.inc"
#include "detect.inc"
#undef MODEL
#define MODEL(sym) {#sym, _binary_##sym##_espdl_start},
static const Entry MODELS[] = {
#include "models.inc"
    {nullptr, nullptr}};
static const Entry DETECT[] = {
#include "detect.inc"
    {nullptr, nullptr}};
#undef MODEL

struct Frame {
    const char *name;
    const uint8_t *rgb;
    float box[4];
    float presence;
};
#define FRAME(sym, x1, y1, x2, y2, p) extern const uint8_t _binary_##sym##_rgb_start[];
#include "frames.inc"
#undef FRAME
#define FRAME(sym, x1, y1, x2, y2, p) {#sym, _binary_##sym##_rgb_start, {x1, y1, x2, y2}, p},
static const Frame FRAMES[] = {
#include "frames.inc"
    {nullptr, nullptr, {0, 0, 0, 0}, 0}};
#undef FRAME
struct Jpeg {
    const char *sym;
    const char *name;
    const uint8_t *start;
    const uint8_t *end;
    float truth[4];
    float desktop[4];
};
#define JPG(sym, name, t1, t2, t3, t4, d1, d2, d3, d4)                                             \
    extern const uint8_t _binary_##sym##_jpg_start[];                                              \
    extern const uint8_t _binary_##sym##_jpg_end[];
#include "jpegs.inc"
#undef JPG
#define JPG(sym, name, t1, t2, t3, t4, d1, d2, d3, d4)                                             \
    {#sym, name, _binary_##sym##_jpg_start, _binary_##sym##_jpg_end, {t1, t2, t3, t4}, {d1, d2, d3, d4}},
static const Jpeg JPEGS[] = {
#include "jpegs.inc"
    {nullptr, nullptr, nullptr, nullptr, {0, 0, 0, 0}, {0, 0, 0, 0}}};
#undef JPG
#ifndef DETECT_SIZE
#define DETECT_SIZE 0
#define DETECT_GAIN 1.0f
#endif

static constexpr int RUNS = 20;
static constexpr int INTERNAL_FIRST = 300 * 1024;
static constexpr float BOX_TOL = 2.0f / 255.0f; // two steps of the uint8 box codec
static constexpr float PRESENCE_TOL = 0.25f;    // logit; decisions sit far from 0

static float bench(dl::Model *model, dl::runtime_mode_t mode)
{
    model->run(mode); // warm-up
    int64_t start = esp_timer_get_time();
    for (int i = 0; i < RUNS; i++) {
        model->run(mode);
    }
    return (float)(esp_timer_get_time() - start) / RUNS / 1000.0f;
}

static void sweep_one(const Entry &e, int max_internal)
{
    size_t int0 = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t ps0 = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    dl::Model *model =
        new dl::Model((const char *)e.data, fbs::MODEL_LOCATION_IN_FLASH_RODATA, max_internal);
    size_t int_used = int0 - heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t ps_used = ps0 - heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

    bool ok = model->test() == ESP_OK; // output matches the values ESP-PPQ exported
    float single_ms = bench(model, dl::RUNTIME_MODE_SINGLE_CORE);
    float dual_ms = bench(model, dl::RUNTIME_MODE_MULTI_CORE);
    printf("BENCH {\"model\":\"%s\",\"max_internal\":%d,\"test\":%s,\"single_ms\":%.2f,"
           "\"dual_ms\":%.2f,\"internal_bytes\":%u,\"psram_bytes\":%u}\n",
           e.name, max_internal, ok ? "true" : "false", single_ms, dual_ms, (unsigned)int_used,
           (unsigned)ps_used);
    delete model;
}

// uint8 RGB (HWC) -> the model's int8 input: real value pixel/255, scale 2^exponent.
static void fill_input(dl::TensorBase *input, const uint8_t *rgb)
{
    static int8_t lut[256];
    static int lut_exponent = 1000;
    int exponent = input->exponent;
    if (exponent != lut_exponent) {
        float inv_scale = ldexpf(1.0f, -exponent);
        for (int v = 0; v < 256; v++) {
            float q = roundf((float)v / 255.0f * inv_scale);
            lut[v] = (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
        }
        lut_exponent = exponent;
    }
    int8_t *dst = (int8_t *)input->data;
    int n = input->get_size();
    for (int i = 0; i < n; i++) {
        dst[i] = lut[rgb[i]];
    }
}

// Marginal soft-argmax of the int8 edge heatmap (NHWC, C = x1,y1,x2,y2) -> normalized xyxy.
static void decode_box(dl::TensorBase *heat, float gain, float box[4])
{
    int h = heat->shape[1], w = heat->shape[2];
    const int8_t *q = (const int8_t *)heat->data;
    float scale = ldexpf(1.0f, heat->exponent) * gain;
    for (int c = 0; c < 4; c++) {
        int8_t peak = -128;
        for (int i = 0; i < h * w; i++) {
            if (q[i * 4 + c] > peak) peak = q[i * 4 + c];
        }
        // Single precision on purpose: the P4's FPU has no doubles, and a software exp()
        // per cell costs more than the whole network. An int8 cell has 256 possible values,
        // so exp() runs once per value, not once per cell.
        float table[256];
        for (int v = 0; v < 256; v++) {
            table[v] = expf((float)(v - 255) * scale); // index = cell - peak + 255
        }
        float total = 0, moment = 0;
        for (int y = 0; y < h; y++) {
            float row = 0, row_moment = 0;
            for (int x = 0; x < w; x++) {
                float p = table[q[(y * w + x) * 4 + c] - peak + 255];
                row += p;
                row_moment += p * (float)x;
            }
            total += row;
            moment += (c & 1) ? row * (float)y / (float)(h - 1) : row_moment / (float)(w - 1);
        }
        box[c] = moment / total;
    }
}

static void soak(const Entry &e)
{
    dl::Model *model =
        new dl::Model((const char *)e.data, fbs::MODEL_LOCATION_IN_FLASH_RODATA, INTERNAL_FIRST);
    bool self_test = model->test() == ESP_OK;
    dl::TensorBase *input = model->get_inputs().begin()->second;
    dl::TensorBase *heat = nullptr, *presence = nullptr;
    for (auto &kv : model->get_outputs()) {
        (kv.second->get_size() == 1 ? presence : heat) = kv.second;
    }
    int n_frames = 0;
    while (FRAMES[n_frames].name) n_frames++;
    bool shapes_ok = heat && presence && heat->shape.size() == 4 && heat->shape[3] == 4 &&
                     input->get_size() == DETECT_SIZE * DETECT_SIZE * 3;
    printf("SOAK_START {\"model\":\"%s\",\"frames\":%d,\"self_test\":%s,\"shapes_ok\":%s,"
           "\"input_exp\":%d,\"heat\":[%d,%d,%d]}\n",
           e.name, n_frames, self_test ? "true" : "false", shapes_ok ? "true" : "false",
           (int)input->exponent, heat ? heat->shape[1] : 0, heat ? heat->shape[2] : 0,
           heat ? heat->shape[3] : 0);
    if (!shapes_ok || n_frames == 0) {
        return;
    }

    std::vector<uint32_t> first_crc(n_frames, 0);
    uint64_t done = 0, desktop_mismatch = 0, self_mismatch = 0;
    float worst_box = 0, worst_presence = 0;
    for (uint32_t round = 0;; round++) {
        int64_t pre_us = 0, net_us = 0, post_us = 0, total_us = 0, max_us = 0;
        for (int f = 0; f < n_frames; f++) {
            const Frame &fr = FRAMES[f];
            int64_t t0 = esp_timer_get_time();
            fill_input(input, fr.rgb);
            int64_t t1 = esp_timer_get_time();
            model->run(dl::RUNTIME_MODE_MULTI_CORE);
            int64_t t2 = esp_timer_get_time();
            float box[4];
            decode_box(heat, DETECT_GAIN, box);
            float logit = ldexpf((float)*(const int8_t *)presence->data, presence->exponent);
            int64_t t3 = esp_timer_get_time();
            pre_us += t1 - t0;
            net_us += t2 - t1;
            post_us += t3 - t2;
            total_us += t3 - t0;
            if (t3 - t0 > max_us) max_us = t3 - t0;

            uint32_t crc = esp_rom_crc32_le(0, (const uint8_t *)heat->data, heat->get_size());
            crc = esp_rom_crc32_le(crc, (const uint8_t *)presence->data, 1);
            if (round == 0) {
                first_crc[f] = crc;
                printf("FRAME {\"name\":\"%s\",\"box\":[%.4f,%.4f,%.4f,%.4f],\"want\":[%.4f,%.4f,"
                       "%.4f,%.4f],\"presence\":%.3f,\"want_presence\":%.3f}\n",
                       fr.name, box[0], box[1], box[2], box[3], fr.box[0], fr.box[1], fr.box[2],
                       fr.box[3], logit, fr.presence);
            } else if (crc != first_crc[f]) {
                self_mismatch++;
            }
            bool bad = fabsf(logit - fr.presence) > PRESENCE_TOL ||
                       (logit >= 0) != (fr.presence >= 0);
            if (fabsf(logit - fr.presence) > worst_presence) worst_presence = fabsf(logit - fr.presence);
            if (fr.presence >= 0) { // a box only means something when the object is there
                for (int c = 0; c < 4; c++) {
                    float d = fabsf(box[c] - fr.box[c]);
                    if (d > worst_box) worst_box = d;
                    bad = bad || d > BOX_TOL;
                }
            }
            desktop_mismatch += bad;
            done++;
        }
        if (round < 3 || round % 25 == 0) {
            printf("SOAK {\"frames\":%llu,\"desktop_mismatch\":%llu,\"self_mismatch\":%llu,"
                   "\"pre_ms\":%.2f,\"net_ms\":%.2f,\"post_ms\":%.2f,\"frame_ms\":%.2f,\"max_ms\":%.2f,\"worst_box\":%.5f,"
                   "\"worst_presence\":%.4f,\"internal_min_free\":%u,\"psram_min_free\":%u,"
                   "\"uptime_s\":%llu}\n",
                   done, desktop_mismatch, self_mismatch, (float)pre_us / n_frames / 1000.0f,
                   (float)net_us / n_frames / 1000.0f, (float)post_us / n_frames / 1000.0f,
                   (float)total_us / n_frames / 1000.0f, (float)max_us / 1000.0f, worst_box,
                   worst_presence, (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
                   (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM),
                   (unsigned long long)(esp_timer_get_time() / 1000000));
        }
    }
}

// Width and height from a baseline JPEG's frame header; false for progressive or broken files.
static bool jpeg_size(const uint8_t *d, size_t n, int *w, int *h)
{
    size_t i = 2;
    while (i + 9 < n) {
        if (d[i] != 0xFF) {
            i++;
            continue;
        }
        uint8_t m = d[i + 1];
        if (m == 0xC0 || m == 0xC1) {
            *h = (d[i + 5] << 8) | d[i + 6];
            *w = (d[i + 7] << 8) | d[i + 8];
            return *w > 0 && *h > 0;
        }
        if (m == 0xC2 || m == 0xDA) {
            return false;
        }
        if (m == 0xD8 || m == 0x01 || (m >= 0xD0 && m <= 0xD7) || m == 0xFF) {
            i += (m == 0xFF) ? 1 : 2;
            continue;
        }
        i += 2 + ((d[i + 2] << 8) | d[i + 3]);
    }
    return false;
}

// Image files as bytes: JPEG decode -> resize + quantize into the model input -> network ->
// box. One "DET {json}" line per file, with the label and the desktop box next to ours.
static void run_jpegs(const Entry &e)
{
    dl::Model *model =
        new dl::Model((const char *)e.data, fbs::MODEL_LOCATION_IN_FLASH_RODATA, INTERNAL_FIRST);
    dl::TensorBase *heat = nullptr;
    for (auto &kv : model->get_outputs()) {
        if (kv.second->get_size() != 1) heat = kv.second;
    }
    // pixel / 255, stretched to the model's square input (how the detector is trained).
    dl::image::ImagePreprocessor pre(model, {0, 0, 0}, {255, 255, 255});
    printf("JPEG_START {\"model\":\"%s\",\"size\":%d}\n", e.name, DETECT_SIZE);
    // Two passes: the software decoder, then the chip's hardware JPEG decoder.
    for (int hw = 0; hw < 2; hw++) {
        for (const Jpeg *j = JPEGS; j->name; j++) {
            dl::image::jpeg_img_t file = {.data = (void *)j->start, .data_len = (size_t)(j->end - j->start)};
            int w = 0, h = 0;
            bool baseline = jpeg_size(j->start, file.data_len, &w, &h);
            int64_t t0 = esp_timer_get_time();
            dl::image::img_t img = {};
            if (baseline) { // neither decoder takes progressive files
                img = hw ? dl::image::hw_decode_jpeg(file, dl::image::DL_IMAGE_PIX_TYPE_RGB888, 500)
                         : dl::image::sw_decode_jpeg(file, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
            }
            int64_t t1 = esp_timer_get_time();
            if (!img.data) {
                printf("DET {\"name\":\"%s\",\"decoder\":\"%s\",\"ok\":false,\"baseline\":%s,"
                       "\"bytes\":%u}\n",
                       j->name, hw ? "hw" : "sw", baseline ? "true" : "false", (unsigned)file.data_len);
                continue;
            }
            // The hardware decoder pads to whole 16-pixel blocks: crop back to the picture.
            pre.preprocess(img, {0, 0, w, h});
            int64_t t2 = esp_timer_get_time();
            model->run(dl::RUNTIME_MODE_MULTI_CORE);
            int64_t t3 = esp_timer_get_time();
            float box[4];
            decode_box(heat, DETECT_GAIN, box);
            int64_t t4 = esp_timer_get_time();
            printf("DET {\"name\":\"%s\",\"decoder\":\"%s\",\"ok\":true,\"bytes\":%u,\"w\":%d,"
                   "\"h\":%d,\"decode_ms\":%.2f,\"resize_ms\":%.2f,\"net_ms\":%.2f,\"post_ms\":%.2f,"
                   "\"box\":[%.4f,%.4f,%.4f,%.4f],\"truth\":[%.4f,%.4f,%.4f,%.4f],"
                   "\"desktop\":[%.4f,%.4f,%.4f,%.4f]}\n",
                   j->name, hw ? "hw" : "sw", (unsigned)file.data_len, w, h, (t1 - t0) / 1000.0f,
                   (t2 - t1) / 1000.0f, (t3 - t2) / 1000.0f, (t4 - t3) / 1000.0f, box[0], box[1],
                   box[2], box[3], j->truth[0], j->truth[1], j->truth[2], j->truth[3], j->desktop[0],
                   j->desktop[1], j->desktop[2], j->desktop[3]);
            heap_caps_free(img.data);
        }
    }
    printf("JPEG_DONE {\"internal_min_free\":%u,\"psram_min_free\":%u}\n",
           (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));
    delete model;
}

// What tools/deploy.py writes at the start of the "model" partition; the .espdl follows at
// MODEL_OFFSET. Little-endian, 64 bytes.
struct ModelHeader {
    char magic[8]; // "TFMODEL1"
    uint32_t bytes;
    uint32_t crc32; // of the .espdl, zlib polynomial
    uint32_t size;  // square input, px
    float gain;     // soft-argmax inverse temperature
    char name[40];
};
static_assert(sizeof(ModelHeader) == 64, "header layout is shared with tools/deploy.py");
static constexpr size_t MODEL_OFFSET = 4096;
static constexpr size_t IMG_LINE_MAX = 3 * 1024 * 1024; // one base64 JPEG per line

// In-place base64 decode; stops at the first byte that is not base64. Returns the length.
static size_t base64_decode(uint8_t *buf, size_t n)
{
    uint32_t acc = 0;
    int bits = 0;
    size_t out = 0;
    for (size_t i = 0; i < n; i++) {
        uint8_t c = buf[i];
        int v = (c >= 'A' && c <= 'Z')   ? c - 'A'
                : (c >= 'a' && c <= 'z') ? c - 'a' + 26
                : (c >= '0' && c <= '9') ? c - '0' + 52
                : (c == '+')             ? 62
                : (c == '/')             ? 63
                                         : -1;
        if (v < 0) break;
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            buf[out++] = (uint8_t)(acc >> bits);
        }
    }
    return out;
}

static void serve_jpeg(dl::Model *model, dl::image::ImagePreprocessor &pre, dl::TensorBase *heat,
                       dl::TensorBase *presence, float gain, uint8_t *data, size_t len)
{
    int w = 0, h = 0;
    if (!jpeg_size(data, len, &w, &h)) {
        printf("DET {\"ok\":false,\"error\":\"not a baseline JPEG\",\"bytes\":%u}\n", (unsigned)len);
        return;
    }
    dl::image::jpeg_img_t file = {.data = (void *)data, .data_len = len};
    int64_t t0 = esp_timer_get_time();
    dl::image::img_t img = dl::image::sw_decode_jpeg(file, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
    int64_t t1 = esp_timer_get_time();
    if (!img.data) {
        printf("DET {\"ok\":false,\"error\":\"decode failed\",\"bytes\":%u}\n", (unsigned)len);
        return;
    }
    pre.preprocess(img, {0, 0, w, h});
    int64_t t2 = esp_timer_get_time();
    model->run(dl::RUNTIME_MODE_MULTI_CORE);
    int64_t t3 = esp_timer_get_time();
    float box[4];
    decode_box(heat, gain, box);
    float logit = ldexpf((float)*(const int8_t *)presence->data, presence->exponent);
    int64_t t4 = esp_timer_get_time();
    printf("DET {\"ok\":true,\"bytes\":%u,\"w\":%d,\"h\":%d,\"decode_ms\":%.2f,\"resize_ms\":%.2f,"
           "\"net_ms\":%.2f,\"post_ms\":%.2f,\"presence\":%.3f,\"box\":[%.4f,%.4f,%.4f,%.4f]}\n",
           (unsigned)len, w, h, (t1 - t0) / 1000.0f, (t2 - t1) / 1000.0f, (t3 - t2) / 1000.0f,
           (t4 - t3) / 1000.0f, logit, box[0], box[1], box[2], box[3]);
    heap_caps_free(img.data);
}

// The deployed model: read from flash, checked, then one detection per image line.
static void serve()
{
    const esp_partition_t *part =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    const void *mapped = nullptr;
    esp_partition_mmap_handle_t handle;
    if (!part || esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &mapped,
                                    &handle) != ESP_OK) {
        printf("MODEL {\"ok\":false,\"error\":\"no model partition\"}\n");
        return;
    }
    ModelHeader hd;
    memcpy(&hd, mapped, sizeof hd);
    const uint8_t *blob = (const uint8_t *)mapped + MODEL_OFFSET;
    if (memcmp(hd.magic, "TFMODEL1", 8) != 0 || hd.bytes == 0 ||
        hd.bytes > part->size - MODEL_OFFSET) {
        printf("MODEL {\"ok\":false,\"error\":\"no model written\"}\n");
        return;
    }
    uint32_t crc = esp_rom_crc32_le(0, blob, hd.bytes);
    hd.name[sizeof hd.name - 1] = 0;
    if (crc != hd.crc32) {
        printf("MODEL {\"ok\":false,\"error\":\"checksum mismatch\",\"crc32\":\"%08x\"}\n",
               (unsigned)crc);
        return;
    }
    dl::Model *model =
        new dl::Model((const char *)blob, fbs::MODEL_LOCATION_IN_FLASH_RODATA, INTERNAL_FIRST);
    bool self_test = model->test() == ESP_OK;
    dl::TensorBase *input = model->get_inputs().begin()->second;
    dl::TensorBase *heat = nullptr, *presence = nullptr;
    for (auto &kv : model->get_outputs()) {
        (kv.second->get_size() == 1 ? presence : heat) = kv.second;
    }
    bool shapes_ok = heat && presence && heat->shape.size() == 4 && heat->shape[3] == 4 &&
                     input->get_size() == (int)(hd.size * hd.size * 3);
    float net_ms = shapes_ok ? bench(model, dl::RUNTIME_MODE_MULTI_CORE) : 0;
    bool ok = self_test && shapes_ok;
    printf("MODEL {\"ok\":%s,\"name\":\"%s\",\"size\":%u,\"gain\":%.4f,\"bytes\":%u,"
           "\"crc32\":\"%08x\",\"self_test\":%s,\"shapes_ok\":%s,\"net_ms\":%.2f,"
           "\"internal_free\":%u,\"psram_free\":%u}\n",
           ok ? "true" : "false", hd.name, (unsigned)hd.size, hd.gain, (unsigned)hd.bytes,
           (unsigned)crc, self_test ? "true" : "false", shapes_ok ? "true" : "false", net_ms,
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    if (!ok) {
        return;
    }
    dl::image::ImagePreprocessor pre(model, {0, 0, 0}, {255, 255, 255});
    uint8_t *line = (uint8_t *)heap_caps_malloc(IMG_LINE_MAX, MALLOC_CAP_SPIRAM);
    uart_driver_install(UART_NUM_0, 32768, 0, 0, nullptr, 0);
    printf("READY\n");
    size_t n = 0;
    bool overflow = false;
    for (;;) {
        uint8_t chunk[1024];
        int got = uart_read_bytes(UART_NUM_0, chunk, sizeof chunk, pdMS_TO_TICKS(20));
        for (int i = 0; i < got; i++) {
            if (chunk[i] != '\n' && chunk[i] != '\r') {
                if (n < IMG_LINE_MAX) {
                    line[n++] = chunk[i];
                } else {
                    overflow = true;
                }
                continue;
            }
            if (overflow) {
                printf("DET {\"ok\":false,\"error\":\"line too long\"}\n");
            } else if (n >= 4 && memcmp(line, "IMG ", 4) == 0) {
                size_t len = base64_decode(line + 4, n - 4);
                serve_jpeg(model, pre, heat, presence, hd.gain, line + 4, len);
            } else if (n >= 4 && n < 64 && memcmp(line, "PING", 4) == 0) {
                // Echo the caller's token, so it can tell this answer from the boot READY.
                printf("READY%.*s\n", (int)(n - 4), (const char *)line + 4);
            } else if (n > 0) {
                printf("DET {\"ok\":false,\"error\":\"unknown command\"}\n");
            }
            n = 0;
            overflow = false;
        }
    }
}

extern "C" void app_main(void)
{
    printf("BENCH_START internal_free=%u psram_free=%u\n",
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    for (const Entry *e = MODELS; e->name; e++) {
        sweep_one(*e, 0);              // activations in PSRAM
        sweep_one(*e, INTERNAL_FIRST); // internal RAM first, spill to PSRAM
    }
    printf("BENCH_DONE\n");
    if (DETECT[0].name && JPEGS[0].name) {
        run_jpegs(DETECT[0]);
    }
    if (DETECT[0].name) {
        soak(DETECT[0]);
    }
    serve();
}
