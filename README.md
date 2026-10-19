# DeepCal AI – Gemini-powered food scanner, diet planner & tracker

## Setup (VS Code terminal)
```bash
cd deepcal
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env           # Mac/Linux: cp .env.example .env
```
Open `.env`, paste your key from https://aistudio.google.com/apikey  and set a long random SECRET_KEY.

## Run
```bash
uvicorn app.main:app --reload --port 8080
```
Open http://localhost:8080 → Sign up → Scan food.

## Features
- Login / signup (hashed passwords, signed tokens, per-user data in SQLite)
- Photo upload, phone-camera capture, and live camera detection (manual or auto-scan every 6 s)
- Multi-item detection: Indian + global dishes, packaged food (reads labels), drinks, fruits
- Editable grams → nutrition recalculates; health score & suggestions
- Dashboard: calorie ring, macros, 7-day chart, today's meals
- BMI, BMR, calorie/macro targets, water goal, AI 7-day diet plan (veg/non-veg, conditions)

## Notes
- Change `GEMINI_MODEL` in `.env` if Google renames/retires the model.
- Camera needs `localhost` or https. Free Gemini keys have rate limits, so keep auto-scan at 6 s or slower.
