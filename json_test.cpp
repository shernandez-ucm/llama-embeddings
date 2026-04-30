#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <Eigen/Dense>
#include <json/json.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>

typedef float Scalar;
const int Dim = Eigen::Dynamic;

struct LatamCitiesQA {
    Eigen::Matrix<Scalar, Dim, 1> embeddings;
    std::map<std::string, std::string> metadata;
};

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
    //LOG_INF("%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
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

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EMBEDDING)) {
        return 1;
    }

    common_init();

    params.embedding = true;

    // get max number of sequences per batch
    //const int n_seq_max = llama_max_parallel_sequences();

    const int n_seq_max = 1;

    // utilize the full context
    if (params.n_batch < params.n_ctx) {
        LOG_WRN("%s: setting batch size to %d\n", __func__, params.n_ctx);
        params.n_batch = params.n_ctx;
    }
    LOG("%s: Batch Size: %i\n", __func__, params.n_batch);
    LOG("%s: Context Size: %i\n", __func__, params.n_ctx);
    
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
    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx       = llama_n_ctx(ctx);

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    if (llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        LOG_ERR("%s: computing embeddings in encoder-decoder models is not supported\n", __func__);
        return 1;
    }

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    std::vector<LatamCitiesQA> dataset;
    std::ifstream file("dataset/latam_cities_1000.jsonl");
    if (!file.is_open()) {
        LOG_ERR("%s: failed to open dataset/latam_cities_10.jsonl\n", __func__);
        return 1;
    }
        

    std::string line;
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errs;

    std::vector<std::string> prompts;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
        if (reader->parse(line.c_str(), line.c_str() + line.length(), &root, &errs)) {
            LatamCitiesQA item;
            item.metadata["question"] = root.get("question", "").asString();
            item.metadata["city"] = root.get("city", "").asString();
            item.metadata["country"] = root.get("country", "").asString();
            item.metadata["topic"] = root.get("topic", "").asString();
            
            if (root.isMember("metadata") && root["metadata"].isObject()) {
                for (auto const& id : root["metadata"].getMemberNames()) {
                    item.metadata[id] = root["metadata"][id].asString();
                }
            }

            prompts.push_back(root.get("answer", "").asString());
            dataset.push_back(item);
        } else {
            LOG_WRN("%s: failed to parse JSON line: %s\n", __func__, errs.c_str());
        }
    }
    const uint64_t n_batch = params.n_batch;
    const std::string added_sep_token = llama_vocab_get_add_sep(vocab) ? llama_vocab_get_text(vocab, llama_vocab_sep(vocab)) : "";
    const std::string added_eos_token = llama_vocab_get_add_eos(vocab) ? llama_vocab_get_text(vocab, llama_vocab_eos(vocab)) : "";
    
    // tokenize the prompts and trim
    std::vector<std::vector<int32_t>> inputs;
    
    for (const auto & prompt : prompts) {
        std::vector<llama_token> inp;
        inp = common_tokenize(ctx, prompt, true, true);
        if (inp.size() > n_batch) {
            LOG_ERR("%s: number of tokens in input line (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                    __func__, (long long int) inp.size(), (long long int) n_batch);
            return 1;
        }
        inputs.push_back(inp);
    }
 
    // initialize batch
    const int n_prompts = prompts.size();
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    // count number of embeddings
    int n_embd_count = n_prompts;

    // allocate output
    const int n_embd_out = llama_model_n_embd_out(model);
    std::vector<float> embeddings(n_embd_count * n_embd_out, 0);
    float * emb = embeddings.data();

    // break into batches
    int e = 0; // number of embeddings already stored
    int s = 0; // number of prompts in current batch
    for (int k = 0; k < n_prompts; k++) {
        // clamp to n_batch tokens
        auto & inp = inputs[k];

        const uint64_t n_toks = inp.size();

        // encode if at capacity
        if (batch.n_tokens + n_toks > n_batch || s >= n_seq_max) {
            float * out = emb + e * n_embd_out;
            batch_decode(ctx, batch, out, s, n_embd_out, params.embd_normalize);
            e += pooling_type == LLAMA_POOLING_TYPE_NONE ? batch.n_tokens : s;
            s = 0;
            common_batch_clear(batch);
        }

        // add to batch
        batch_add_seq(batch, inp, s);
        s += 1;
    }
    
    /*for (auto & item : dataset) {
        LOG("%s: question: %s\n", __func__, item.metadata["question"].c_str());
        LOG("%s: city: %s\n", __func__, item.metadata["city"].c_str());
        LOG("%s: country: %s\n", __func__, item.metadata["country"].c_str());
        LOG("%s: topic: %s\n", __func__, item.metadata["topic"].c_str());
        LOG("%s: answer: %s\n", __func__, answers[&item - &dataset[0]].c_str());
        LOG("\n");
    }*/
    for (int i = 0; i < (int)dataset.size(); ++i) {
        dataset[i].embeddings.resize(n_embd_out);
        for (int j = 0; j < n_embd_out; ++j) {
            dataset[i].embeddings(j) = embeddings[i * n_embd_out + j];
        }
    }
    LOG("\n");
    //llama_perf_context_print(ctx);

    // clean up
    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
