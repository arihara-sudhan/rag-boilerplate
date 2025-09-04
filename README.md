# RAG (Retrieval-Augmented Generation) Application

A simple yet powerful RAG system that allows you to ask questions about content stored in a corpus and get AI-generated answers based on the retrieved context.

## Features

- **Document Chunking**: Automatically splits large documents into manageable chunks with overlap
- **Vector Search**: Uses FAISS (Facebook AI Similarity Search) for efficient similarity search
- **Embeddings**: Leverages SentenceTransformers for high-quality text embeddings
- **AI Generation**: Powered by Google's Gemini 2.5 Flash model for generating contextual answers
- **Persistent Storage**: Saves FAISS index and chunk mappings for faster subsequent runs

## Prerequisites

- Python 3.7+
- Google Gemini API key

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install python-dotenv numpy faiss-cpu google-generativeai sentence-transformers
```

3. Create a `.env` file in the project root and add your Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```

## Usage

1. **Prepare your corpus**: Place your text content in `corpus.txt`
2. **Run the application**:

```bash
python main.py
```

3. **Ask questions**: The system will prompt you to enter questions about your corpus content
4. **Exit**: Type 'Q' to quit the application

## How It Works

### 1. Document Processing
- Reads content from `corpus.txt`
- Splits text into chunks (default: 200 words with 50-word overlap)
- Generates embeddings for each chunk using SentenceTransformers

### 2. Index Creation
- Creates a FAISS index for efficient similarity search
- Saves the index (`faiss_index.bin`) and chunk mappings (`id_to_chunk.pkl`) for reuse

### 3. Query Processing
- Converts your question to an embedding
- Searches for the most similar chunks in the corpus
- Retrieves top-k most relevant chunks as context

### 4. Answer Generation
- Passes the retrieved context and your question to Gemini
- Generates a contextual answer based on the corpus content

## File Structure

```
rag/
├── main.py              # Main application file
├── corpus.txt           # Your text corpus
├── faiss_index.bin      # FAISS index (auto-generated)
├── id_to_chunk.pkl      # Chunk mappings (auto-generated)
├── .env                 # Environment variables (create this)
└── README.md           # This file
```

## Configuration

You can modify these parameters in `main.py`:

- **Chunk size**: Change `chunk_size` in `chunk_text()` function (default: 200 words)
- **Overlap**: Change `overlap` in `chunk_text()` function (default: 50 words)
- **Top-k results**: Change `top_k` in `rag_query()` function (default: 3 chunks)
- **Embedding model**: Change the model in `SentenceTransformer()` (default: 'all-MiniLM-L6-v2')

## Example

```
Enter your question (or type Q to quit): What happened to Ari and Shakthi?

--- RAG Answer ---

Based on the context provided, Ari and Shakthi were trapped in different restrooms at CIT College Campus, Coimbatore on Saturday, Aug 23, 2025, because there was no water available. After Shakthi finished using the restroom, he discovered there was no water for flushing or using the jet spray. Panicked, he contacted Sharvesh to check what Ari was doing to resolve the situation. Ari then realized there was no water available and urged Sharvesh and Deva to bring water in a water can. The embarrassed guys brought water into the restroom, which was crowded with students, and a very small amount of water helped resolve their situation.
```

## Troubleshooting

- **API Key Issues**: Ensure your `GEMINI_API_KEY` is correctly set in the `.env` file
- **Memory Issues**: For large corpora, consider reducing chunk size or using a smaller embedding model
- **Slow Performance**: The first run creates the index and may take longer; subsequent runs will be faster

## Dependencies

- `python-dotenv`: Environment variable management
- `numpy`: Numerical operations
- `faiss-cpu`: Vector similarity search
- `google-generativeai`: Google Gemini API client
- `sentence-transformers`: Text embeddings

## License

This project is open source and available under the MIT License.
