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

static void batch_decode(llama_context * ctx, llama_batch & batch, float * output, int /*n_seq*/, int n_embd_out, int embd_norm) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    llama_memory_clear(llama_get_memory(ctx), true);

    if (llama_decode(ctx, batch) < 0) {
        LOG_ERR("%s : failed to process\n", __func__);
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        const float * embd = nullptr;
        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            embd = llama_get_embeddings_ith(ctx, i);
        } else {
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        }

        int embd_pos = (pooling_type == LLAMA_POOLING_TYPE_NONE) ? i : batch.seq_id[i][0];
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

    if (params.n_parallel == 1) {
        params.kv_unified = true;
        params.n_parallel = llama_max_parallel_sequences();
    }

    if (params.n_batch < params.n_ctx) {
        params.n_batch = params.n_ctx;
    }

    if (params.attention_type != LLAMA_ATTENTION_TYPE_CAUSAL) {
        params.n_ubatch = params.n_batch;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd_out = llama_model_n_embd_out(model);
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    std::vector<LatamCitiesQA> dataset;
    std::ifstream file("dataset/latam_cities.jsonl");
    std::string line;
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errs;

    std::vector<std::string> answers;

    while (std::getline(file, line)) {
        std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
        if (reader->parse(line.c_str(), line.c_str() + line.length(), &root, &errs)) {
            LatamCitiesQA item;
            item.metadata["question"] = root["question"].asString();
            item.metadata["city"] = root["city"].asString();
            item.metadata["country"] = root["country"].asString();
            item.metadata["topic"] = root["topic"].asString();
            
            if (root.isMember("metadata") && root["metadata"].isObject()) {
                for (auto const& id : root["metadata"].getMemberNames()) {
                    item.metadata[id] = root["metadata"][id].asString();
                }
            }

            answers.push_back(root["answer"].asString());
            dataset.push_back(item);
        }
    }

    LOG_INF("%s: Loaded %zu items from dataset\n", __func__, dataset.size());

    std::vector<std::vector<int32_t>> inputs;
    for (const auto & answer : answers) {
        inputs.push_back(common_tokenize(vocab, answer, true, true));
    }

    const int n_prompts = inputs.size();
    struct llama_batch batch = llama_batch_init(params.n_batch, 0, 1);

    std::vector<float> all_embeddings(n_prompts * n_embd_out, 0);
    float * emb_data = all_embeddings.data();

    int e = 0; // number of embeddings already stored
    int s = 0; // number of prompts in current batch
    for (int k = 0; k < n_prompts; k++) {
        auto & inp = inputs[k];
        if (batch.n_tokens + (int)inp.size() > (int)params.n_batch || s >= (int)params.n_parallel) {
            float * out = emb_data + e * n_embd_out;
            batch_decode(ctx, batch, out, s, n_embd_out, params.embd_normalize);
            e += (pooling_type == LLAMA_POOLING_TYPE_NONE) ? batch.n_tokens : s;
            s = 0;
            common_batch_clear(batch);
        }
        batch_add_seq(batch, inp, s);
        s += 1;
    }

    if (batch.n_tokens > 0) {
        float * out = emb_data + e * n_embd_out;
        batch_decode(ctx, batch, out, s, n_embd_out, params.embd_normalize);
    }

    // Transfer to Eigen
    for (int i = 0; i < n_prompts; ++i) {
        dataset[i].embeddings.resize(n_embd_out);
        for (int j = 0; j < n_embd_out; ++j) {
            dataset[i].embeddings(j) = all_embeddings[i * n_embd_out + j];
        }
    }

    LOG_INF("%s: Processed %zu embeddings\n", __func__, dataset.size());

    // Simple verification
    if (!dataset.empty()) {
        std::cout << "\nExample Metadata for first item:\n";
        for (auto const& [key, val] : dataset[0].metadata) {
            std::cout << "  " << key << ": " << val << "\n";
        }
        std::cout << "Embedding (first 5 elements): " << dataset[0].embeddings.head(5).transpose() << "\n";
    }

    llama_batch_free(batch);
    llama_backend_free();

    return 0;
}
