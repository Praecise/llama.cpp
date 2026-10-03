// Gradient-only training on the serving graph: finite-difference checks of llama_opt_grad_sequence
// on a small random dense model, through the KV cache with and without flash attention, plus
// accumulation, reset, repeatability across thread counts and the refusal codes.

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const uint32_t N_VOCAB = 48;
static const uint32_t N_EMBD  = 32;
static const uint32_t N_HEAD  = 2;
static const uint32_t N_FF    = 48;
static const uint32_t N_LAYER = 2;
static const uint32_t N_CTX   = 64;

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    seed ^= std::hash<std::string>{}(tensor->name);
    std::mt19937 gen(seed);
    const bool norm = strstr(tensor->name, "norm") != nullptr;
    std::normal_distribution<float> dis(norm ? 1.0f : 0.0f, norm ? 0.1f : 0.2f);

    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    std::vector<float> tmp(ggml_nelements(tensor));
    for (float & x : tmp) {
        x = dis(gen);
    }
    ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
}

static gguf_context_ptr model_metadata() {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_LLAMA, ret.get());
    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,         llm_arch_name(LLM_ARCH_LLAMA));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                   N_VOCAB);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,               N_CTX);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,             N_EMBD);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                  N_LAYER);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,          N_FF);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,         N_HEAD);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,      uint32_t(1)); // grouped-query heads
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,  1e-5f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,         N_EMBD/N_HEAD);
    ms.add_kv(LLM_KV_ROPE_FREQ_BASE,               10000.0f);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,              "no_vocab");
    return ret;
}

static bool select_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

static llama_context_ptr make_ctx(llama_model * model, bool flash_attn, ggml_type type_kv, int n_threads) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = N_CTX;
    cp.n_batch         = N_CTX;
    cp.n_ubatch        = N_CTX;
    cp.n_seq_max       = 1;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.type_k          = type_kv;
    cp.type_v          = type_kv;
    cp.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    return llama_context_ptr(llama_init_from_model(model, cp));
}

// -sum_rows sum(targets * log_softmax(logits)) / n_tokens, in double
static double ce_loss(const std::vector<float> & logits, const std::vector<float> & targets, int n_tokens) {
    double loss = 0.0;
    for (int i = 0; i < n_tokens; ++i) {
        const float * l = logits.data() + (size_t) i*N_VOCAB;
        const float * t = targets.data() + (size_t) i*N_VOCAB;
        double mx = l[0];
        for (uint32_t j = 1; j < N_VOCAB; ++j) {
            mx = std::max(mx, (double) l[j]);
        }
        double sum = 0.0;
        for (uint32_t j = 0; j < N_VOCAB; ++j) {
            sum += std::exp(l[j] - mx);
        }
        const double lse = mx + std::log(sum);
        for (uint32_t j = 0; j < N_VOCAB; ++j) {
            loss -= t[j] * (l[j] - lse);
        }
    }
    return loss / n_tokens;
}

static std::vector<float> read_f32(const ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

static int n_fail = 0;

static void check(bool ok, const char * what) {
    printf("  %-60s %s\n", what, ok ? "OK" : "FAIL");
    n_fail += ok ? 0 : 1;
}

static void test_fd(llama_model * model, bool flash_attn) {
    printf("finite differences, flash attention %s\n", flash_attn ? "on" : "off");
    llama_context_ptr lctx = make_ctx(model, flash_attn, GGML_TYPE_F32, 4);
    GGML_ASSERT(lctx);
    check(llama_opt_grad_init(lctx.get(), model, select_all, nullptr) == 0, "init");
    check(llama_opt_grad_init(lctx.get(), model, select_all, nullptr) == -1, "second init refused");

    const int n_tokens = 12;
    std::mt19937 gen(7);
    std::vector<llama_token> tokens(n_tokens);
    std::vector<float> targets((size_t) n_tokens*N_VOCAB, 0.0f);
    for (int i = 0; i < n_tokens; ++i) {
        tokens[i] = gen() % N_VOCAB;
        targets[(size_t) i*N_VOCAB + gen() % N_VOCAB] = 1.0f;
    }
    std::vector<float> logits((size_t) n_tokens*N_VOCAB);

    check(llama_opt_grad_sequence(lctx.get(), tokens.data(), n_tokens, targets.data(), logits.data()) == 0, "backward pass");
    const double loss0 = ce_loss(logits, targets, n_tokens);

    {
        std::vector<std::string> ops;
        for (int32_t i = 0; i < llama_opt_grad_n_ops(lctx.get()); ++i) {
            ops.push_back(llama_opt_grad_op(lctx.get(), i));
        }
        auto has = [&](const char * op) { return std::find(ops.begin(), ops.end(), op) != ops.end(); };
        check(std::is_sorted(ops.begin(), ops.end()) && std::adjacent_find(ops.begin(), ops.end()) == ops.end() &&
              has("MUL_MAT") && has("OUT_PROD") && has("CROSS_ENTROPY_LOSS_BACK") && has("SET_ROWS") &&
              llama_opt_grad_op(lctx.get(), -1) == nullptr, "graph op list");
    }

    const char * names[] = {
        "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.1.attn_output.weight",
        "blk.1.ffn_up.weight", "blk.0.attn_norm.weight", "output.weight",
    };
    for (const char * name : names) {
        ggml_tensor * w = llama_model_get_tensor(model, name);
        GGML_ASSERT(w);
        ggml_tensor * g = llama_opt_grad(lctx.get(), w);
        if (!g) {
            check(false, name);
            continue;
        }
        const std::vector<float> grad = read_f32(g);
        std::vector<float> data = read_f32(w);

        // the 6 largest-magnitude gradient entries plus 2 fixed ones
        std::vector<size_t> idx;
        {
            std::vector<size_t> order(grad.size());
            for (size_t i = 0; i < order.size(); ++i) {
                order[i] = i;
            }
            std::partial_sort(order.begin(), order.begin() + 6, order.end(),
                [&](size_t a, size_t b) { return std::fabs(grad[a]) > std::fabs(grad[b]); });
            idx.assign(order.begin(), order.begin() + 6);
            idx.push_back(0);
            idx.push_back(grad.size()/2);
        }

        double max_err = 0.0;
        double max_g   = 0.0;
        for (size_t i : idx) {
            const float eps = 1e-2f;
            const float x = data[i];
            double l[2];
            for (int s = 0; s < 2; ++s) {
                data[i] = x + (s == 0 ? eps : -eps);
                ggml_backend_tensor_set(w, data.data(), 0, ggml_nbytes(w));
                GGML_ASSERT(llama_opt_grad_sequence(lctx.get(), tokens.data(), n_tokens, nullptr, logits.data()) == 0);
                l[s] = ce_loss(logits, targets, n_tokens);
            }
            data[i] = x;
            ggml_backend_tensor_set(w, data.data(), 0, ggml_nbytes(w));
            const double fd = (l[0] - l[1]) / (2.0*eps);
            max_err = std::max(max_err, std::fabs(fd - grad[i]));
            max_g   = std::max(max_g, std::fabs((double) grad[i]));
        }
        char what[128];
        snprintf(what, sizeof(what), "%s (max |g| %.3e, max err %.1e)", name, max_g, max_err);
        check(max_g > 1e-5 && max_err <= 2e-3*std::max(1.0, max_g) + 1e-4, what);
    }

    // forward-only passes leave the gradients alone; a second backward pass adds the same amount
    ggml_tensor * wk = llama_model_get_tensor(model, "blk.0.attn_k.weight");
    const std::vector<float> g1 = read_f32(llama_opt_grad(lctx.get(), wk));
    GGML_ASSERT(llama_opt_grad_sequence(lctx.get(), tokens.data(), n_tokens, targets.data(), logits.data()) == 0);
    check(std::fabs(ce_loss(logits, targets, n_tokens) - loss0) < 1e-6, "forward unchanged after the probes");
    const std::vector<float> g2 = read_f32(llama_opt_grad(lctx.get(), wk));
    bool doubled = true;
    for (size_t i = 0; i < g1.size(); ++i) {
        doubled = doubled && std::fabs(g2[i] - 2.0f*g1[i]) <= 1e-6f*(1.0f + std::fabs(g1[i]));
    }
    check(doubled, "gradients accumulate");

    llama_opt_grad_reset(lctx.get());
    bool zero = true;
    for (float x : read_f32(llama_opt_grad(lctx.get(), wk))) {
        zero = zero && x == 0.0f;
    }
    check(zero, "reset zeroes the gradients");

    check(llama_opt_grad_sequence(lctx.get(), tokens.data(), N_CTX + 1, targets.data(), nullptr) == -2, "sequence longer than the ubatch refused");
}

static void test_repeatable(llama_model * model) {
    printf("repeatability across thread counts\n");
    const int n_tokens = 16;
    std::vector<llama_token> tokens(n_tokens);
    std::vector<float> targets((size_t) n_tokens*N_VOCAB, 0.0f);
    for (int i = 0; i < n_tokens; ++i) {
        tokens[i] = (i*7 + 3) % N_VOCAB;
        targets[(size_t) i*N_VOCAB + (i*11 + 5) % N_VOCAB] = 1.0f;
    }
    std::vector<std::vector<float>> runs;
    for (int n_threads : {1, 3, 8}) {
        llama_context_ptr lctx = make_ctx(model, false, GGML_TYPE_F32, n_threads);
        GGML_ASSERT(llama_opt_grad_init(lctx.get(), model, select_all, nullptr) == 0);
        GGML_ASSERT(llama_opt_grad_sequence(lctx.get(), tokens.data(), n_tokens, targets.data(), nullptr) == 0);
        std::vector<float> all;
        for (const char * name : {"blk.0.attn_k.weight", "blk.1.ffn_down.weight", "output.weight"}) {
            const std::vector<float> g = read_f32(llama_opt_grad(lctx.get(), llama_model_get_tensor(model, name)));
            all.insert(all.end(), g.begin(), g.end());
        }
        runs.push_back(all);
    }
    bool same = true;
    for (size_t r = 1; r < runs.size(); ++r) {
        same = same && memcmp(runs[0].data(), runs[r].data(), runs[0].size()*sizeof(float)) == 0;
    }
    check(same, "bitwise identical gradients with 1, 3 and 8 threads");
}

static void test_refusals(llama_model * model) {
    printf("refusals\n");
    llama_context_ptr lctx = make_ctx(model, false, GGML_TYPE_F16, 4);
    check(llama_opt_grad_init(lctx.get(), model, select_all, nullptr) == -3, "F16 KV cache refused");
    std::vector<llama_token> tokens = {1, 2, 3};
    check(llama_opt_grad_sequence(lctx.get(), tokens.data(), 3, nullptr, nullptr) == -1, "pass without init refused");

    llama_context_ptr lctx2 = make_ctx(model, false, GGML_TYPE_F32, 4);
    auto none = [](const struct ggml_tensor *, void *) { return false; };
    check(llama_opt_grad_init(lctx2.get(), model, none, nullptr) == -4, "empty parameter selection refused");
}

int main(void) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    llama_backend_init();
    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fputs(text, stderr);
        }
    }, nullptr);

    gguf_context_ptr meta = model_metadata();
    size_t seed = 1234;
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t devs[] = {cpu, nullptr};
    mp.devices = devs;
    llama_model_ptr model(llama_model_init_from_user(meta.get(), set_tensor_data, &seed, mp));
    GGML_ASSERT(model);

    test_refusals(model.get());
    test_fd(model.get(), false);
    test_fd(model.get(), true);
    test_repeatable(model.get());

    llama_backend_free();
    printf("%s\n", n_fail == 0 ? "PASSED" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
