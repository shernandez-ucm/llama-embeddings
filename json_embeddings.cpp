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
#include <algorithm>

typedef float Scalar;
const int Dim = Eigen::Dynamic;

struct LatamCitiesQA {
    Eigen::Matrix<Scalar, Dim, 1> embeddings;
    std::map<std::string, std::string> metadata;
};

// --- Helper Functions ---

static void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, llama_seq_id seq_id) {
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, { seq_id }, true);
    }
}

static void batch_decode(llama_context * ctx, llama_batch & batch, float * output, int n_embd_out, int embd_norm) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    llama_memory_clear(llama_get_memory(ctx), true);

    if (llama_decode(ctx, batch) < 0) {
        LOG_ERR("%s : failed to process\n", __func__);
        return;
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        const float * embd = nullptr;
        int embd_pos = 0;

        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            embd = llama_get_embeddings_ith(ctx, i);
            embd_pos = i;
        } else {
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            embd_pos = batch.seq_id[i][0];
        }

        if (embd == nullptr) {
            LOG_ERR("%s : failed to get embeddings for token/seq %d\n", __func__, embd_pos);
            continue;
        }

        float * out = output + embd_pos * n_embd_out;
        common_embd_normalize(embd, out, n_embd_out, embd_norm);
    }
}

// --- Implementation Functions ---

std::vector<LatamCitiesQA> load_latam_dataset(const std::string & filename, std::vector<std::string> & questions) {
    std::vector<LatamCitiesQA> dataset;
    std::ifstream file(filename);
    if (!file.is_open()) {
        LOG_ERR("%s: failed to open %s\n", __func__, filename.c_str());
        return dataset;
    }

    std::string line;
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errs;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
        if (reader->parse(line.c_str(), line.c_str() + line.length(), &root, &errs)) {
            LatamCitiesQA item;
            item.metadata["question"] = root.get("question", "").asString();
            item.metadata["city"] = root.get("city", "").asString();
            item.metadata["country"] = root.get("country", "").asString();
            item.metadata["topic"] = root.get("topic", "").asString();
            item.metadata["answer"] = root.get("answer", "").asString(); // Keep answer for display
            
            if (root.isMember("metadata") && root["metadata"].isObject()) {
                for (auto const& id : root["metadata"].getMemberNames()) {
                    item.metadata[id] = root["metadata"][id].asString();
                }
            }

            questions.push_back(root.get("question", "").asString());
            dataset.push_back(item);
        } else {
            LOG_WRN("%s: failed to parse JSON line: %s\n", __func__, errs.c_str());
        }
    }

    LOG_INF("%s: Loaded %zu items from %s\n", __func__, dataset.size(), filename.c_str());
    return dataset;
}

void compute_embeddings(
    const llama_model * model,
    llama_context * ctx,
    const common_params & params,
    const std::vector<std::string> & texts,
    std::vector<Eigen::Matrix<Scalar, Dim, 1>> & output_embeddings
) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd_out = llama_model_n_embd_out(model);
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);

    std::vector<std::vector<int32_t>> inputs;
    for (const auto & text : texts) {
        inputs.push_back(common_tokenize(ctx, text, true, true));
    }

    for (size_t i = 0; i < inputs.size(); i++) {
        if (inputs[i].size() > params.n_batch) {
            LOG_ERR("%s: input %zu exceeds batch size (%zu > %d)\n", __func__, i, inputs[i].size(), (int)params.n_batch);
            return;
        }
    }

    int n_embd_count = (pooling_type == LLAMA_POOLING_TYPE_NONE) ? 0 : (int)inputs.size();
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        for (const auto & inp : inputs) n_embd_count += (int)inp.size();
    }

    std::vector<float> embeddings_buffer(n_embd_count * n_embd_out, 0);
    float * emb_ptr = embeddings_buffer.data();

    struct llama_batch batch = llama_batch_init(params.n_batch, 0, 1);
    const int n_seq_max = llama_max_parallel_sequences();

    int e = 0; // number of embeddings already stored
    int s = 0; // number of prompts in current batch
    for (int k = 0; k < (int)inputs.size(); k++) {
        auto & inp = inputs[k];
        if (batch.n_tokens + (int)inp.size() > (int)params.n_batch || s >= n_seq_max) {
            float * out = emb_ptr + e * n_embd_out;
            batch_decode(ctx, batch, out, n_embd_out, params.embd_normalize);
            e += (pooling_type == LLAMA_POOLING_TYPE_NONE) ? batch.n_tokens : s;
            s = 0;
            common_batch_clear(batch);
        }
        batch_add_seq(batch, inp, s);
        s += 1;
    }

    if (batch.n_tokens > 0) {
        float * out = emb_ptr + e * n_embd_out;
        batch_decode(ctx, batch, out, n_embd_out, params.embd_normalize);
    }

    if (pooling_type != LLAMA_POOLING_TYPE_NONE) {
        output_embeddings.resize(texts.size());
        for (int i = 0; i < (int)texts.size(); ++i) {
            output_embeddings[i].resize(n_embd_out);
            for (int j = 0; j < n_embd_out; ++j) {
                output_embeddings[i](j) = embeddings_buffer[i * n_embd_out + j];
            }
        }
    }

    llama_batch_free(batch);
}

// --- Main ---

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

    // 1. Load Dataset
    std::vector<std::string> questions;
    std::vector<LatamCitiesQA> dataset = load_latam_dataset("dataset/latam_cities_1000.jsonl", questions);

    if (dataset.empty()) {
        LOG_ERR("%s: no data to process\n", __func__);
        return 1;
    }

    // 2. Compute Embeddings for dataset
    std::vector<Eigen::Matrix<Scalar, Dim, 1>> dataset_embeddings;
    compute_embeddings(model, ctx, params, questions, dataset_embeddings);
    
    for (size_t i = 0; i < dataset.size(); ++i) {
        dataset[i].embeddings = dataset_embeddings[i];
    }

    LOG_INF("%s: Dataset processed. Ready for queries.\n", __func__);

    // 3. Interactive Query Loop
    std::string query;
    while (true) {
        std::cout << "\nEnter query (or 'exit' to quit): ";
        if (!std::getline(std::cin, query) || query == "exit") {
            break;
        }
        if (query.empty()) continue;

        std::vector<Eigen::Matrix<Scalar, Dim, 1>> query_emb_vec;
        compute_embeddings(model, ctx, params, {query}, query_emb_vec);

        if (query_emb_vec.empty()) continue;
        const auto & query_emb = query_emb_vec[0];

        // Calculate similarities
        std::vector<std::pair<float, int>> similarities;
        for (int i = 0; i < (int)dataset.size(); ++i) {
            // Since vectors are normalized, cosine similarity is just the dot product
            float sim = query_emb.dot(dataset[i].embeddings);
            similarities.push_back({sim, i});
        }

        // Sort by similarity descending
        std::sort(similarities.begin(), similarities.end(), 
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
            });

        // Print top 3 matches
        std::cout << "\nTop Matches:\n";
        for (int i = 0; i < std::min(3, (int)similarities.size()); ++i) {
            int idx = similarities[i].second;
            float score = similarities[i].first;
            std::cout << i + 1 << ". [Score: " << score << "] " 
                      << dataset[idx].metadata["city"] << " (" << dataset[idx].metadata["country"] << ")\n";
            std::cout << "   Q: " << dataset[idx].metadata["question"] << "\n";
            // Print a snippet of the answer
            std::string ans = dataset[idx].metadata["answer"];
            if (ans.length() > 200) ans = ans.substr(0, 197) + "...";
            std::cout << "   A: " << ans << "\n\n";
        }
    }

    // 4. Cleanup
    llama_backend_free();

    return 0;
}