from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import os
import tempfile
from dotenv import load_dotenv

# Charger les variables depuis .env
load_dotenv()

app = Flask(__name__)
CORS(app)

# Clé Groq — même que dans server.js
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Modèle Groq
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
)

# Embeddings locaux gratuits
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = None

@app.route("/rag/upload", methods=["POST"])
def upload_document():
    global vectorstore

    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier envoyé"}), 400

    file = request.files["file"]
    filename = file.filename.lower()
    suffix = ".pdf" if filename.endswith(".pdf") else ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        os.unlink(tmp_path)

        return jsonify({
            "success": True,
            "message": f"Document chargé : {len(chunks)} sections indexées.",
            "filename": file.filename,
        })

    except Exception as e:
        os.unlink(tmp_path)
        return jsonify({"error": str(e)}), 500


@app.route("/rag/ask", methods=["POST"])
def ask_document():
    global vectorstore

    if vectorstore is None:
        return jsonify({"error": "Aucun document chargé."}), 400

    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Question vide"}), 400

    try:
        # Récupérer les passages pertinents
        docs = vectorstore.similarity_search(question, k=4)
        context = "\n\n".join([d.page_content for d in docs])

        # Construire le prompt manuellement
        prompt = f"""Tu es KPC IA. Réponds à la question en te basant uniquement sur le contexte fourni.
Si la réponse n'est pas dans le contexte, dis-le clairement.
Réponds en français.

Contexte :
{context}

Question : {question}

Réponse :"""

        response = llm.invoke(prompt)
        return jsonify({"reply": response.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/rag/status", methods=["GET"])
def status():
    return jsonify({"loaded": vectorstore is not None})


if __name__ == "__main__":
    print("\n🐍 Serveur RAG démarré sur http://localhost:5000")
    print("📄 Prêt à lire des PDF et fichiers TXT\n")
    app.run(port=5000, debug=False)
