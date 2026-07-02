# TuneBot — Music-Streaming Data Chatbot

TuneBot is a natural-language chatbot that answers questions about a
music-streaming database. Ask it questions in plain English — *"Which artists
have the most plays?"*, *"How many premium users signed up in 2023?"* — and it
translates them into SQL, runs them against a local SQLite database, and replies
with a clear, concise answer.

## How it works

```
user question
     │
     ▼
  LLM agent  ──(decides to query)──►  execute_sql_query tool
     ▲                                        │
     │                                        ▼
     └──────── SQL result (JSON) ────  SQLite (music_streaming.db)
     │
     ▼
natural-language answer
```

- **LLM** —Get your own API key
- **Agent** — a LangChain tool-calling agent (`create_agent`, LangChain 1.x /
  LangGraph) that loops *model → tool → model* until it has a final answer.
- **Tool** — a single callable, `execute_sql_query`, that runs **read-only** SQL
  against the local SQLite database and returns rows as JSON.

If the model writes bad SQL, the error is handed back to it so it can correct
itself and retry automatically. Conversation history is retained in-session, so
follow-up questions keep their context.

## Safety & guardrails

TuneBot is built to be read-only and on-topic:

- **Read-only** — only single `SELECT` / `WITH` statements are allowed. Any
  `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `TRUNCATE`, `PRAGMA`,
  etc. is rejected before it ever reaches the database.
- **Single statement** — semicolon-chained queries are blocked.
- **On-topic** — the system prompt instructs the bot to politely decline
  questions unrelated to the music-streaming data.
- **Privacy** — the bot prefers aggregates and won't reveal its system prompt or
  connection details.

## Database schema

The SQLite database (`music_streaming.db`) has five tables:

| Table     | Key columns                                                              |
|-----------|--------------------------------------------------------------------------|
| `artists` | `artist_id`, `name`, `genre`, `country`, `formed_year`                   |
| `albums`  | `album_id`, `artist_id →`, `title`, `release_year`, `label`              |
| `tracks`  | `track_id`, `album_id →`, `title`, `duration_seconds`, `explicit`        |
| `users`   | `user_id`, `name`, `email`, `country`, `subscription_tier`, `signup_date`|
| `plays`   | `play_id`, `user_id →`, `track_id →`, `played_at`, `ms_played`           |

A "play" is one listening event; `ms_played` is how long the user actually
listened (a value below the track length implies a skip).

## Setup

1. **Create/activate a virtual environment** (optional but recommended) and
   install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables.** Create a `.env` file in the project root:

   ```ini
   AZURE_OPENAI_API_KEY=your-epam-dial-api-key
   AZURE_OPENAI_ENDPOINT=https://ai-proxy.lab.epam.com
   OPENAI_API_VERSION=2025-04-01-preview
   DEPLOYMENT_MODEL=gpt-4o
   ```

   Only `AZURE_OPENAI_API_KEY` is strictly required; the others have sensible
   defaults.

3. **Build the database** (run once — re-running rebuilds it from scratch):

   ```bash
   python init_db.py
   ```

## Run

```bash
python chatbot.py
```

You'll get an interactive prompt:

```
TuneBot - your music-streaming data assistant.
Ask a question in plain English. Type 'exit' or 'quit' to leave.

You: which 5 artists have the most total plays?
TuneBot: ...
```

Type `exit` or `quit` (or press Ctrl+C) to leave.

## Example questions

- "How many tracks are longer than 5 minutes?"
- "Which genre has the most plays overall?"
- "List the top 5 users by number of plays."
- "How many premium subscribers signed up in 2023?"
- "What's the average track duration per album for artist X?"

## Files

| File               | Purpose                                             |
|--------------------|-----------------------------------------------------|
| `chatbot.py`       | The chatbot — agent, SQL tool, and interactive loop |
| `init_db.py`       | Builds `music_streaming.db` from the SQL schema     |
| `requirements.txt` | Python dependencies                                 |
| `.env`             | API credentials and model configuration (not committed) |
