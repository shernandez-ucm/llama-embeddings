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

int main(int argc, char ** argv) {
    
    std::vector<LatamCitiesQA> dataset;
    std::ifstream file("dataset/latam_cities_10.jsonl");
    if (!file.is_open()) {
        LOG_ERR("%s: failed to open dataset/latam_cities_10.jsonl\n", __func__);
        return 1;
    }

    std::string line;
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errs;

    std::vector<std::string> answers;

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

            answers.push_back(root.get("answer", "").asString());
            dataset.push_back(item);
        } else {
            LOG_WRN("%s: failed to parse JSON line: %s\n", __func__, errs.c_str());
        }
    }

    for (auto & item : dataset) {
        LOG("%s: question: %s\n", __func__, item.metadata["question"].c_str());
        LOG("%s: city: %s\n", __func__, item.metadata["city"].c_str());
        LOG("%s: country: %s\n", __func__, item.metadata["country"].c_str());
        LOG("%s: topic: %s\n", __func__, item.metadata["topic"].c_str());
        LOG("%s: answer: %s\n", __func__, answers[&item - &dataset[0]].c_str());
        LOG("\n");
    }
    return 0;
}
