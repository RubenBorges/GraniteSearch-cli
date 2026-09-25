#pragma once
#ifndef GRANITE_EMBEDDER_HPP
#define GRANITE_EMBEDDER_HPP

#include <string>
#include <vector>

#include <llama.h>

class GraniteEmbedder {
private:
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const int EMBEDDING_DIM = 384;

public:
    // Default parameter sits safely inside header declaration layer
    GraniteEmbedder(const std::string& model_path, int max_ctx = 512);
    ~GraniteEmbedder();

    std::vector<float> compute_embedding(const std::string& text);
    int get_dimension() const;
};

namespace granite{

// Simple, effective rule-based sentence chunker for C++
std::vector<std::string> split_into_sentences(const std::string& text) ;

// Calculates the Cosine Similarity between two 384-dimensional arrays
double calculate_cosine_similarity(const std::vector<float>& vecA, const std::vector<float>& vecB) ;
// Struct to store un-embedded text metadata pairs
struct RawChunk {
    std::string text;
    std::string source_doc; 
};

// Struct to guarantee sentences and vectors never lose alignment
struct EmbeddedChunk {
    std::string text;
    std::string source_doc; 
    std::vector<float> embedding;
};

struct SearchResult {
    double score;
    std::string text;
    std::string source_doc; 
};

} // namespace granite

#endif