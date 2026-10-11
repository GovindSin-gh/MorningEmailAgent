# Personalized Morning-Email Agent

A LangGraph agent that researches current weather, Indian history for today's date, yesterday's Indian news, retrieves personal context from FAISS RAG, builds an HTML email, and can send it through Gmail.

## Files

- `run.py` — complete agent script, including tool setup and LangGraph workflow.
- `requirements.txt` — Python dependencies.
- `.github/workflows/morning-email.yml` — scheduled GitHub Actions workflow.

## Before first run

1. Put the required FAISS index files in `faiss_index/` at the repository root. They must match the embedding model used to create the index.
2. In GitHub: **Settings → Secrets and variables → Actions → Secrets**, add:
   - `GROQ_API_KEY`
   - `TAVILY_API_KEY`
   - `GOOGLE_API_KEY`
   - `GMAIL` — Gmail sender address
   - `GOOGLE_APP_PASS` — Gmail App Password (not your normal Gmail password)
3. Under **Variables**, add:
   - `CITY` = `Lucknow` (or your chosen city)
   - `REGION` = `Uttar Pradesh`
   - `COUNTRY` = `India`
   - `LAT` = `26.8467`
   - `LON` = `80.9462`
   - `RECIPIENT_EMAIL` = recipient address
   - `SENDER_NAME` = `govinds-ai-agent`
   - `DRY_RUN` = `true` initially
4. Keep `.env` out of Git. Add `.env` and any personal source documents to `.gitignore`.
5. Confirm `faiss_index/` does not expose private documents. Prefer a private repository if the index contains personal data.

## Schedule

GitHub Actions cron is interpreted in UTC. `30 18 * * *` runs at 18:30 UTC, equivalent to midnight at the start of the next day in India (IST, UTC+5:30). Scheduled runs can be delayed by GitHub and are not guaranteed to start at the exact second. The workflow also supports manual runs from the Actions tab.

## First test

1. Push these files to the repository's default branch.
2. Open **Actions → Personalized Morning Email → Run workflow**.
3. With `DRY_RUN=true`, inspect the log and generated HTML. No email should be sent.
4. Once the content and RAG evidence look right, change the repository variable `DRY_RUN` to `false`.
5. Run manually again and verify delivery. Future scheduled runs will then send email.

## Notes

- Dates are calculated in Asia/Kolkata time.
- Location comes from explicit variables, not the GitHub runner's IP address.
- Never print API keys or Gmail credentials in logs.
- The FAISS index is loaded with `allow_dangerous_deserialization=True`; only use an index you created or otherwise trust.
"# MorningEmailAgent" 
