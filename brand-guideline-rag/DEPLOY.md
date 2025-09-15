# Deployment

**Live URL:** test-ai-pm-portfolio-mqhyd3rghuapp539c3zuu3t

## How it’s deployed
- Hosting: Streamlit Community Cloud
- Source: This folder in the `main` branch
- Entry point: `brand-guideline-rag/app.py`
- Dependencies: `requirements.txt` at repo root

## Update flow
1) Edit files in GitHub (or locally).
2) Commit changes to `main`.
3) Streamlit auto-redeploys. Check build logs in the Streamlit dashboard.

## Secrets (later, if needed)
- In the app’s Streamlit dashboard → **Settings → Secrets**
- Add items like:
