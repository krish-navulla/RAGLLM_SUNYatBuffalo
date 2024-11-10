
# LLM-Enhanced Chatbot with Retrieval-Augmented Generation (RAG) for Assisting University at Buffalo Students, for the Department of Computer Science

This project implements a **chatbot application** powered by a **Large Language Model (LLM)** using **Retrieval-Augmented Generation (RAG)**. The chatbot provides intelligent answers based on content from the **University at Buffalo Computer Science and Engineering** webpage and its subpages. The project leverages LangChain, FAISS for vector storage, OpenAI embeddings, and Streamlit for the user interface.

## 🔗 Data Source
The main content used for this project is extracted from the [University at Buffalo Computer Science and Engineering webpage](https://engineering.buffalo.edu/computer-science-engineering.html), including its subpages.

## 🛠️ Features
- **Web Scraping**: Automatically extracts content from the main page and its subpages using `WebBaseLoader`.
- **Text Chunking**: Splits the extracted text into manageable chunks for efficient processing.
- **Vector Storage**: Uses FAISS for storing and retrieving vector embeddings.
- **LLM Integration**: Uses GPT-4o model with OpenAI embeddings for intelligent question-answering.
- **Persistent Index**: Caches the FAISS index to avoid repeated scraping and indexing.
- **Streamlit UI**: Provides an interactive user interface for querying the chatbot.

## 🚀 Tech Stack
- **Python**
- **LangChain**: Document loaders, embeddings, and RAG components
- **FAISS**: Efficient vector search
- **OpenAI**: Embeddings and LLM (GPT-4o)
- **Streamlit**: User interface
- **BeautifulSoup**: HTML parsing
- **PyYAML**: Secure API key management

## 📦 Installation

### Prerequisites
- Python 3.8+
- `conda` or `venv` for environment management

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/llm-chatbot.git
   cd llm-chatbot
   ```

2. **Create a Virtual Environment**:
   ```bash
   conda create -n llmchatbot python=3.11
   conda activate llmchatbot
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `secrets.yaml` File**:
   In the project directory, create a `secrets.yaml` file to store your OpenAI API key:
   ```yaml
   OPENAI_API_KEY: "your-openai-api-key-here"
   ```

5. **Add `secrets.yaml` to `.gitignore`**:
   Ensure your API key is not exposed:
   ```bash
   echo "secrets.yaml" >> .gitignore
   ```

## 🏃‍♂️ Running the Application
Start the Streamlit app using:
```bash
streamlit run app.py
```

### Access the App
- Local URL: [http://localhost:8501](http://localhost:8501)
- Network URL: Your network IP (useful for sharing on the local network)

## 🧩 Project Structure
```graphql
llm-chatbot/
├── app.py                # Main application file
├── secrets.yaml          # API key configuration (ignored in git)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Ignored files
├── faiss_index/          # Directory for storing the FAISS index
```

## 📋 Key Functions
- **Web Scraping**:
  - Extracts content from the main page and its subpages using `WebBaseLoader`.
- **Indexing**:
  - Creates a FAISS index for efficient retrieval of relevant content.
- **Query Handling**:
  - Uses the FAISS index and GPT-4o embeddings for context-aware responses.

## 🔍 Example Queries
Try asking the chatbot:
- "What health and wellness resources are available for students?"
- "How can faculty members access community support services?"
- "What is the procedure for students of concern at UB?"

## ⚠️ Troubleshooting
1. **API Key Error**:
   - Ensure your `OPENAI_API_KEY` is correctly set in `secrets.yaml`.
   - Verify the key is loaded with:
     ```python
     print(os.environ.get("OPENAI_API_KEY"))
     ```

2. **FAISS Index Not Found**:
   - The index will be created on the first run. If you encounter issues, delete the `faiss_index` folder and restart the app.

3. **Deprecation Warnings**:
   - Use the LangChain CLI to update imports:
     ```bash
     langchain upgrade imports
     ```

## 🤖 Future Enhancements
- **Multi-page Web Scraping**: Extend the crawler to handle additional domains.
- **Advanced Caching**: Store the FAISS index in a database for distributed usage.
- **Improved UI**: Add additional features to the Streamlit interface for enhanced user experience.

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## ✨ Acknowledgments
- [LangChain Documentation](https://python.langchain.com/docs/versions/v0_2/)
- [FAISS Library](https://github.com/facebookresearch/faiss)
- [OpenAI](https://openai.com/) for LLM support
- University at Buffalo for providing the data source
