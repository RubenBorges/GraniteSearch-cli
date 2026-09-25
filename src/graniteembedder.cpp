// graniteembedder.cpp

#include <cmath>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include <graniteembedder.hpp>

// Constructor handles initialization and loading
// Fix: Default argument (= 512) moved entirely to header file declaration
GraniteEmbedder::GraniteEmbedder(const std::string& model_path, int max_ctx) {
    // 1. Initialize global system backend
    llama_backend_init();

    // 2. Load the model parameters and files safely
    llama_model_params model_params = llama_model_default_params();
    model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        throw std::runtime_error("Error: Failed to load model file from: " + model_path);
    }

    // 3. Configure the modern embedding context setup
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = max_ctx;
    ctx_params.embeddings = true;
    
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_CLS;

    ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        llama_model_free(model);
        throw std::runtime_error("Error: Failed to initialize llama context from model.");
    }
}

// Destructor manages strict memory tracking and safety loops (RAII)
GraniteEmbedder::~GraniteEmbedder() {
    if (ctx) {
        llama_free(ctx);
    }
    if (model) {
        llama_model_free(model);
    }
    llama_backend_free();
}

// Discrete method to extract a raw float vector from text strings
std::vector<float> GraniteEmbedder::compute_embedding(const std::string& text) {
    if (!ctx || !model) {
        throw std::runtime_error("Embedder error: Model/Context not properly initialized.");
    }

    // 4. Clean tokenization sequence loop
    std::vector<llama_token> tokens(text.size() + 2);
    
    // Get a pointer to the model's vocabulary mapping system
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (!vocab) {
        throw std::runtime_error("Embedder error: Failed to extract vocabulary from model.");
    }

    // Pass 'vocab' as the first parameter instead of 'model'
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), true, true);
    }
    tokens.resize(n_tokens);

    if (tokens.empty()) {
        return std::vector<float>(EMBEDDING_DIM, 0.0f);
    }

    // 5. Build execution batch graph allocations 
    // Instantly create a standard evaluation batch out of your continuous token memory
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

    // Encode context for this encoder-only embedding model.
    if (llama_encode(ctx, batch) != 0) {
        throw std::runtime_error("Inference Engine error: Failed to process text encoding step.");
    }
    // 6. Access the specific embedding array location for your batch directly
    // This bypasses the strict sequence mapping requirement
    float* raw_embeddings = llama_get_embeddings_seq(ctx, 0);

    if (!raw_embeddings) {
        raw_embeddings = llama_get_embeddings_ith(ctx, 0);
    }

    if (!raw_embeddings) {
        throw std::runtime_error("Extraction error: Engine failed to retrieve processed embedding address.");
    }

    // Convert the raw pointer output into a safe C++ standard vector wrapper
    std::vector<float> embedding_vector(raw_embeddings, raw_embeddings + EMBEDDING_DIM);

    return embedding_vector;
}

int GraniteEmbedder::get_dimension() const { return EMBEDDING_DIM; }


// Simple, effective rule-based sentence chunker for C++
std::vector<std::string> granite::split_into_sentences(const std::string& text) {
    std::vector<std::string> sentences;
    std::regex sentence_regex("[^.!?]+([.!?]+|$)\\s*");
    
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), sentence_regex);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        std::string str = match.str();
        
        str.erase(str.find_last_not_of(" \n\r\t") + 1);
        if (!str.empty()) {
            sentences.push_back(str);
        }
    }
    return sentences;
};

// Calculates the Cosine Similarity between two 384-dimensional arrays
double granite::calculate_cosine_similarity(const std::vector<float>& vecA, const std::vector<float>& vecB) {
    if (vecA.size() != vecB.size()) {
        throw std::invalid_argument("Vectors must be of the same dimension.");
    }

    double dot_product = 0.0;
    double magnitude_A = 0.0;
    double magnitude_B = 0.0;

    for (size_t i = 0; i < vecA.size(); ++i) {
        dot_product += static_cast<double>(vecA[i] * vecB[i]);
        magnitude_A += static_cast<double>(vecA[i] * vecA[i]);
        magnitude_B += static_cast<double>(vecB[i] * vecB[i]);
    }

    magnitude_A = std::sqrt(magnitude_A);
    magnitude_B = std::sqrt(magnitude_B);

    if (magnitude_A == 0.0 || magnitude_B == 0.0) {
        return 0.0;
    }

    return dot_product / (magnitude_A * magnitude_B);
};

