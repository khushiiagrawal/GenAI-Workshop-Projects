# GenAI Workshop Projects

A collection of hands-on Generative AI projects built with open-source models, designed for educational workshops and learning purposes. These projects demonstrate practical applications of AI without requiring API keys or paid services.

## 🚀 Projects Overview

This repository contains three main projects that showcase different aspects of Generative AI:

### 1. Personal AI Assistant (`Personal_AI_Assistant.ipynb`)
A conversational AI assistant built using Google's Flan-T5 model that can engage in interactive chat sessions.

**Key Features:**
- No API key required - uses open-source models
- Interactive chat interface
- Chat history tracking
- Context-aware responses
- Simple setup for Google Colab

**Technologies Used:**
- Hugging Face Transformers
- Google Flan-T5 Small model
- Python

### 2. Document Q&A System (`FileQA_opensourcemodel_langchain_day_2_(1).ipynb`)
A document question-answering system that allows users to upload documents and ask questions about their content.

**Key Features:**
- Support for multiple file formats (PDF, DOCX, TXT)
- Document chunking and processing
- Vector-based similarity search
- Interactive Q&A interface
- Retrieval-augmented generation (RAG)

**Technologies Used:**
- LangChain framework
- FAISS vector database
- Hugging Face Embeddings
- Google Flan-T5 Base model
- Sentence Transformers

### 3. Startup Idea Generator (`Idea_Generato_day3ripynb.ipynb`)
A creative AI tool that generates startup ideas and pitch content based on domain inputs.

**Key Features:**
- Domain-specific startup idea generation
- Structured pitch creation (Problem, Solution, Business Idea)
- Interactive Gradio web interface
- Shareable web application
- Creative content generation

**Technologies Used:**
- Hugging Face Transformers
- MBZUAI LaMini-Flan-T5-783M model
- Gradio for web interface
- Python

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- Google Colab (recommended) or Jupyter Notebook
- Internet connection for model downloads

### Installation

#### For Personal AI Assistant:
```bash
pip install transformers torch accelerate
```

#### For Document Q&A System:
```bash
pip install langchain langchain-community
pip install faiss-cpu
pip install pypdf python-docx
pip install sentence-transformers
pip install transformers
```

#### For Startup Idea Generator:
```bash
pip install transformers gradio torch accelerate
```

## 📖 Usage Guide

### Personal AI Assistant
1. Open the notebook in Google Colab
2. Run all cells to install dependencies and load the model
3. Start chatting with the assistant
4. Type 'exit' to end the conversation

### Document Q&A System
1. Open the notebook in Google Colab
2. Install required dependencies
3. Upload a document (PDF, DOCX, or TXT)
4. Wait for document processing and vector store creation
5. Ask questions about your document content
6. Type 'exit' to quit

### Startup Idea Generator
1. Open the notebook in Google Colab
2. Install dependencies and load the model
3. Launch the Gradio interface
4. Enter a domain (e.g., "Agritech", "Fintech", "EdTech")
5. Get generated startup ideas and pitches
6. Share the public URL with others

## 🎯 Learning Objectives

These projects are designed to teach:

- **Open-source AI models**: Working with Hugging Face Transformers
- **Document processing**: Text chunking, embeddings, and vector search
- **RAG (Retrieval-Augmented Generation)**: Combining retrieval with generation
- **Interactive AI applications**: Building user-friendly interfaces
- **Creative AI**: Using AI for content generation and ideation
- **Web deployment**: Creating shareable AI applications

## 🔧 Technical Details

### Models Used
- **Google Flan-T5 Small/Base**: Instruction-tuned text generation models
- **MBZUAI LaMini-Flan-T5-783M**: Enhanced instruction-following model
- **sentence-transformers/all-MiniLM-L6-v2**: Embedding model for document search

### Key Libraries
- **Transformers**: Hugging Face's library for transformer models
- **LangChain**: Framework for building LLM applications
- **FAISS**: Facebook's library for efficient similarity search
- **Gradio**: Python library for creating ML web apps
- **PyPDF/Docx2txt**: Document processing libraries

## 🌟 Features Highlights

- **No API Keys Required**: All projects use free, open-source models
- **Google Colab Ready**: Optimized for cloud-based execution
- **Educational Focus**: Clear explanations and step-by-step implementation
- **Interactive Interfaces**: User-friendly chat and web interfaces
- **Multiple File Formats**: Support for various document types
- **Shareable Applications**: Gradio apps can be shared publicly

