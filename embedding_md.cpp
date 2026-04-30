#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <ctime>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static std::string read_file(const std::string & fname) {
    std::ifstream file(fname, std::ios::binary);
    if (!file) {
        throw std::runtime_error("error: failed to open file " + fname);
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return content;
}

static void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, llama_seq_id seq_id) {
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, { seq_id }, true);
    }
}

static void batch_decode(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd_out, int embd_norm) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    // clear previous kv_cache values (irrelevant for embeddings)
    llama_memory_clear(llama_get_memory(ctx), true);

    // run model
    LOG_INF("%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
    if (llama_decode(ctx, batch) < 0) {
        LOG_ERR("%s : failed to process\n", __func__);
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        const float * embd = nullptr;
        int embd_pos = 0;

        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // try to get token embeddings
            embd = llama_get_embeddings_ith(ctx, i);
            embd_pos = i;
            GGML_ASSERT(embd != NULL && "failed to get token embeddings");
        } else {
            // try to get sequence embeddings - supported only when pooling_type is not NONE
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            embd_pos = batch.seq_id[i][0];
            GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");
        }

        float * out = output + embd_pos * n_embd_out;
        common_embd_normalize(embd, out, n_embd_out, embd_norm);
    }
}

int main(int argc, char ** argv) {
    common_params params;

    // Use default retrieval params if not specified
    params.chunk_size = 512;
    params.chunk_overlap = 64;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EMBEDDING)) {
        return 1;
    }

    common_init();

    params.embedding = true;

    if (params.context_files.empty()) {
        LOG_ERR("%s: error: no markdown files specified. Use --context-file <path>\n", __func__);
        return 1;
    }

    // get max number of sequences per batch
    const int n_seq_max = llama_max_parallel_sequences();

    if (params.n_parallel == 1) {
        LOG_INF("%s: n_parallel == 1 -> unified KV cache is enabled\n", __func__);
        params.kv_unified = true;
        params.n_parallel = n_seq_max;
    }

    // utilize the full context
    if (params.n_batch < params.n_ctx) {
        LOG_WRN("%s: setting batch size to %d\n", __func__, params.n_ctx);
        params.n_batch = params.n_ctx;
    }

    // for non-causal models, batch size must be equal to ubatch size
    if (params.attention_type != LLAMA_ATTENTION_TYPE_CAUSAL) {
        params.n_ubatch = params.n_batch;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx = llama_init->context();

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx       = llama_n_ctx(ctx);
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    // process files
    std::vector<std::vector<llama_token>> inputs;
    for (const auto & filename : params.context_files) {
        LOG_INF("%s: processing file '%s'\n", __func__, filename.c_str());
        std::string content = read_file(filename);
        std::vector<llama_token> tokens = common_tokenize(ctx, content, true, true);

        if (tokens.empty()) {
            continue;
        }

        // chunk tokens
        int chunk_size = params.chunk_size;
        int chunk_overlap = params.chunk_overlap;

        if (chunk_size <= 0) {
            LOG_ERR("%s: chunk_size must be positive\n", __func__);
            return 1;
        }
        if (chunk_overlap >= chunk_size) {
            LOG_WRN("%s: chunk_overlap (%d) is >= chunk_size (%d), setting to chunk_size - 1\n", __func__, chunk_overlap, chunk_size);
            chunk_overlap = chunk_size - 1;
        }

        size_t start = 0;
        while (start < tokens.size()) {
            size_t end = std::min(start + (size_t)chunk_size, tokens.size());
            std::vector<llama_token> chunk(tokens.begin() + start, tokens.begin() + end);
            
            // check if the last token is SEP/EOS, if not and model expects it, maybe add it?
            // common_tokenize already added it if requested. 
            // For chunks, we might want to ensure they are properly terminated if the model needs it.
            
            if (chunk.size() > (size_t)params.n_batch) {
                LOG_ERR("%s: chunk size (%zu) exceeds batch size (%d)\n", __func__, chunk.size(), params.n_batch);
                return 1;
            }
            inputs.push_back(chunk);

            if (end == tokens.size()) break;
            start += (size_t)(chunk_size - chunk_overlap);
        }
    }

    if (inputs.empty()) {
        LOG_ERR("%s: no inputs to process\n", __func__);
        return 1;
    }

    // initialize batch
    const int n_chunks = inputs.size();
    const uint64_t n_batch = params.n_batch;
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // allocate output
    const int n_embd_out = llama_model_n_embd_out(model);
    std::vector<float> embeddings(n_chunks * n_embd_out, 0);
    float * emb = embeddings.data();

    // break into batches
    int e = 0; // number of embeddings already stored
    int s = 0; // number of chunks in current batch
    for (int k = 0; k < n_chunks; k++) {
        auto & inp = inputs[k];
        const uint64_t n_toks = inp.size();

        if (batch.n_tokens + n_toks > n_batch || s >= n_seq_max) {
            float * out = emb + e * n_embd_out;
            batch_decode(ctx, batch, out, s, n_embd_out, params.embd_normalize);
            e += pooling_type == LLAMA_POOLING_TYPE_NONE ? batch.n_tokens : s;
            s = 0;
            common_batch_clear(batch);
        }

        batch_add_seq(batch, inp, s);
        s += 1;
    }

    // final batch
    float * out = emb + e * n_embd_out;
    batch_decode(ctx, batch, out, s, n_embd_out, params.embd_normalize);

    // print summary
    LOG_INF("\nGenerated %d embeddings for %zu chunks\n", n_chunks, inputs.size());
    for (int i = 0; i < n_chunks; i++) {
        LOG("embedding %d (size %zu tokens): ", i, inputs[i].size());
        for (int j = 0; j < std::min(8, n_embd_out); j++) {
            LOG("%9.6f ", emb[i * n_embd_out + j]);
        }
        LOG("...\n");
    }

    // clean up
    llama_batch_free(batch);
    llama_backend_free();

    return 0;
}
