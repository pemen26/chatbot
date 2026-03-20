@echo off
echo Demarrage de KPC IA...
start "RAG Server" cmd /k "python rag_server.py"
timeout /t 2 /nobreak >nul
start "KPC IA" cmd /k "npm run dev"
echo Les deux serveurs sont lances !
echo - Chatbot : http://localhost:3000
echo - RAG     : http://localhost:5000
