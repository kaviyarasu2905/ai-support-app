# 🎯 AI Support Reply Generator

An intelligent multi-industry customer support automation system built with **LangGraph**, **RAG (Retrieval-Augmented Generation)**, and **Groq LLaMA 3.1**.

---

## 🚀 Features

- 🤖 **AI-Powered Replies** — Automatically generates professional support responses using Groq LLaMA 3.1
- 📚 **RAG Knowledge Base** — Retrieves answers from past tickets using FAISS vector search
- 🔴 **Auto Priority** — LLM assigns ticket priority based on past ticket history
- 📧 **Email Fallback** — Automatically sends escalation emails when no KB match is found
- 💬 **Chat History** — View full conversation history in chat format
- 📊 **Live Analytics** — Real-time resolution rate, ratings, and ticket stats
- 📋 **Ticket Logging** — All tickets saved to Excel for download
- 🏢 **8 Industries Supported** — Art of Living, Healthcare, Banking, E-Commerce, Education, Manufacturing, Travel, Real Estate
- 📎 **Screenshot Attachments** — Attach images to tickets and emails

---

## 🏗️ Tech Stack

| Tool | Purpose |
|------|---------|
| Streamlit | Frontend UI |
| LangChain | LLM pipeline |
| LangGraph | Node graph pipeline |
| Groq (LLaMA 3.1) | Language model |
| FAISS | Vector similarity search |
| Sentence Transformers | Embeddings for RAG |
| PyPDF | PDF knowledge base reader |
| OpenPyXL | Excel ticket logging |

---

## ⚙️ Setup & Deployment

### Environment Variables

Set these in your deployment platform (Render / Streamlit Cloud):

| Key | Description |
|-----|-------------|
| `GROQ_API_KEY` | Your Groq API key from [console.groq.com](https://console.groq.com) |
| `SMTP_EMAIL` | Your Gmail address for sending emails |
| `SMTP_PASSWORD` | Your Gmail App Password |

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY=your_key_here
export SMTP_EMAIL=your_email@gmail.com
export SMTP_PASSWORD=your_app_password

# Run the app
streamlit run app.py
```

### Deploy on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect this repository
4. Set **Start Command:**
   ```
   streamlit run app.py --server.port 10000 --server.headless true
   ```
5. Add environment variables in the Render dashboard
6. Click **Deploy**

---

## 📖 How to Use

1. **Upload KB PDF** — Upload your knowledge base PDF from the sidebar
2. **Select Industry** — Choose from 8 supported industries
3. **Fill Ticket Details** — Enter name, email, category, and priority
4. **Describe the Issue** — Type the support issue clearly
5. **Generate Reply** — Click the Generate button to run the AI pipeline
6. **Email Sent Automatically** — Reply is emailed to the user if email is provided

---

## 🔄 AI Pipeline

```
START → Classify → Priority → Retrieve → Generate → Polish → EMAIL
```

| Step | Description |
|------|-------------|
| Classify | Routes ticket to the correct category |
| Priority | LLM assigns priority using past ticket history |
| Retrieve | RAG search finds similar past tickets from KB |
| Generate | Drafts a reply or escalation message |
| Polish | Final quality check on the reply |
| Email | Sends the response to the user |

---

## 📊 Priority Levels

| Priority | SLA |
|----------|-----|
| 🔴 Critical | 2 hours |
| 🟠 High | 4 hours |
| 🟡 Medium | 24 hours |
| 🟢 Low | 48 hours |
| 🤖 Automatic | LLM decides based on history |

---

## 👨‍💻 Author

**Kavi Govintharaj**
Built with ❤️ using LangChain, LangGraph, and Streamlit
