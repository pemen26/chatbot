import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';
import { readFileSync } from 'fs';

// Charger les variables d'environnement depuis .env
try {
  const env = readFileSync('.env', 'utf-8');
  env.split('\n').forEach(line => {
    const [key, ...rest] = line.split('=');
    if (key && rest.length) process.env[key.trim()] = rest.join('=').trim();
  });
} catch {}


const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = 3000;

const GROQ_API_KEY = process.env.GROQ_API_KEY || '';
const GROQ_MODEL = 'llama-3.3-70b-versatile';

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.post('/api/chat', async (req, res) => {
  const { messages, allHistory } = req.body;

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: 'Messages invalides' });
  }

  // Construire le contexte mémoire depuis les conversations précédentes
  let memoryContext = '';
  if (allHistory && allHistory.length > 0) {
    const previousExchanges = allHistory
      .slice(0, 10) // max 10 échanges passés
      .map(m => `${m.role === 'user' ? 'Utilisateur' : 'KPC IA'}: ${m.content}`)
      .join('\n');
    memoryContext = `\n\nVoici un résumé des échanges précédents avec cet utilisateur pour contexte :\n${previousExchanges}\n\nContinue la conversation en tenant compte de ce contexte.`;
  }

  try {
    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${GROQ_API_KEY}`,
      },
      body: JSON.stringify({
        model: GROQ_MODEL,
        messages: [
          {
            role: 'system',
            content: `Tu es KPC IA, un assistant IA intelligent, utile et bienveillant. Réponds toujours en français sauf si l'utilisateur écrit dans une autre langue.${memoryContext}`,
          },
          ...messages,
        ],
        max_tokens: 1024,
        temperature: 0.7,
      }),
    });

    if (!response.ok) {
      const err = await response.json();
      return res.status(response.status).json({ error: err.error?.message || 'Erreur API Groq' });
    }

    const data = await response.json();
    const reply = data.choices?.[0]?.message?.content || '';
    res.json({ reply });

  } catch (error) {
    console.error('Erreur:', error.message);
    res.status(500).json({ error: 'Erreur interne : ' + error.message });
  }
});

app.listen(PORT, () => {
  console.log(`\n🤖 KPC IA démarré sur http://localhost:${PORT}`);
  console.log(`⚡ Propulsé par Groq (${GROQ_MODEL})\n`);
});
