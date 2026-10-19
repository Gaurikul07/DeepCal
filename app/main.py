import os, io, json, time, hmac, hashlib, base64, sqlite3, secrets, datetime
from pathlib import Path
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env", override=True)
KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
print("Using Gemini model:", MODEL)
SECRET = os.getenv("SECRET_KEY", "dev-secret-change-me").encode()
app = FastAPI(title="DeepCal AI")

# ---------------- DB ----------------
def db():
    c = sqlite3.connect(BASE / "deepcal.db"); c.row_factory = sqlite3.Row
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, pw TEXT, salt TEXT, profile TEXT DEFAULT '{}');
    CREATE TABLE IF NOT EXISTS meals(id INTEGER PRIMARY KEY, uid INTEGER, day TEXT, meal TEXT, food TEXT, grams REAL,
      calories REAL, protein REAL, fat REAL, carbs REAL, fiber REAL, sugar REAL, sodium REAL);
    CREATE TABLE IF NOT EXISTS plans(uid INTEGER PRIMARY KEY, plan TEXT);""")
    return c

# ---------------- Auth (PBKDF2 + signed token) ----------------
def hpw(pw, salt): return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 120_000).hex()
def make_token(uid):
    body = f"{uid}.{int(time.time()) + 7*86400}"
    return body + "." + hmac.new(SECRET, body.encode(), "sha256").hexdigest()
def current_user(authorization: str = Header(None)):
    try:
        uid, exp, sig = (authorization or "").replace("Bearer ", "").split(".")
        ok = hmac.compare_digest(sig, hmac.new(SECRET, f"{uid}.{exp}".encode(), "sha256").hexdigest())
        if not ok or int(exp) < time.time(): raise ValueError
    except Exception:
        raise HTTPException(401, "Please login again")
    with db() as c:
        u = c.execute("SELECT * FROM users WHERE id=?", (int(uid),)).fetchone()
    if not u: raise HTTPException(401, "User not found")
    return dict(u)

class Reg(BaseModel): name: str; email: str; password: str
class Login(BaseModel): email: str; password: str

@app.post("/api/register")
def register(r: Reg):
    if len(r.password) < 6: raise HTTPException(400, "Password must be at least 6 characters")
    salt = secrets.token_hex(8)
    try:
        with db() as c:
            cur = c.execute("INSERT INTO users(name,email,pw,salt) VALUES(?,?,?,?)", (r.name.strip(), r.email.lower().strip(), hpw(r.password, salt), salt))
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Email already registered")
    return {"token": make_token(cur.lastrowid), "name": r.name}

@app.post("/api/login")
def login(r: Login):
    with db() as c:
        u = c.execute("SELECT * FROM users WHERE email=?", (r.email.lower().strip(),)).fetchone()
    if not u or not hmac.compare_digest(u["pw"], hpw(r.password, u["salt"])):
        raise HTTPException(401, "Wrong email or password")
    return {"token": make_token(u["id"]), "name": u["name"]}

@app.get("/api/me")
def me(u=Depends(current_user)):
    return {"name": u["name"], "email": u["email"], "profile": json.loads(u["profile"]), "gemini": bool(KEY)}

# ---------------- Gemini ----------------
def gemini(parts, schema_hint=""):
    if not KEY: raise HTTPException(400, "GEMINI_API_KEY missing. Add it to the .env file and restart the server.")
    body = {"contents": [{"parts": parts}], "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2}}
    models = [MODEL] + [m for m in os.getenv("GEMINI_FALLBACK_MODEL", "").split(",") if m.strip()]
    last = None
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model.strip()}:generateContent"
        for attempt in range(3):
            try:
                r = requests.post(url, headers={"x-goog-api-key": KEY}, json=body, timeout=90)
            except requests.RequestException as e:
                last = str(e); time.sleep(2); continue
            if r.status_code == 429:
                raise HTTPException(429, "Gemini quota/rate limit reached. Auto-scan band karke 1-2 minute ruko (ya kal try karo / billing enable karo).")
            if r.status_code in (500, 503, 504):
                last = f"{r.status_code}"; time.sleep(2 * (attempt + 1)); continue
            if r.status_code != 200:
                last = f"{r.status_code}: {r.text[:200]}"; break   # try next model
            try:
                txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(txt.strip().removeprefix("```json").removesuffix("```"))
            except Exception:
                raise HTTPException(502, "Could not read the AI response, please try again")
    raise HTTPException(503, f"Gemini is busy right now ({last}). Please try again in a minute.")

FOOD_PROMPT = """You are an expert nutritionist and food-recognition system. Look at the image and identify EVERY food or drink item visible
(home-cooked, restaurant, street food, Indian regional dishes like dosa/idli/samosa/biryani/chapati/dal/paneer, fast food, fruits, vegetables,
sweets, beverages, and packaged/branded products). For packaged food, read the brand/label; if a nutrition label is visible use its values.
Estimate each item's weight in grams from plate/bowl/hand/object size. Nutrition values must be for the estimated weight (not per 100 g).
If no food is visible return {"items": [], "note": "No food detected"}.
Return ONLY JSON: {"items":[{"name":str,"category":"home|restaurant|packaged|fruit|beverage|other","grams":number,"calories":number,
"protein":number,"fat":number,"carbs":number,"fiber":number,"sugar":number,"sodium_mg":number,"glycemic_index":number,"confidence":0-100}],
"health_score":number(1-10),"health_note":str,"suggestions":[str,str,str]}"""

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), u=Depends(current_user)):
    data = await file.read()
    mime = file.content_type if (file.content_type or "").startswith("image/") else "image/jpeg"
    return gemini([{"text": FOOD_PROMPT}, {"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}}])

# ---------------- Meals / dashboard ----------------
class Item(BaseModel):
    name: str; grams: float = 0; calories: float = 0; protein: float = 0; fat: float = 0
    carbs: float = 0; fiber: float = 0; sugar: float = 0; sodium_mg: float = 0
class MealIn(BaseModel): meal: str; items: list[Item]

@app.post("/api/meals")
def add_meals(m: MealIn, u=Depends(current_user)):
    day = str(datetime.date.today())
    with db() as c:
        for i in m.items:
            c.execute("INSERT INTO meals(uid,day,meal,food,grams,calories,protein,fat,carbs,fiber,sugar,sodium) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (u["id"], day, m.meal, i.name, i.grams, i.calories, i.protein, i.fat, i.carbs, i.fiber, i.sugar, i.sodium_mg))
    return {"ok": True}

@app.delete("/api/meals/{mid}")
def del_meal(mid: int, u=Depends(current_user)):
    with db() as c: c.execute("DELETE FROM meals WHERE id=? AND uid=?", (mid, u["id"]))
    return {"ok": True}

def calc_plan(p):
    w, h, a = float(p["weight"]), float(p["height"]), int(p["age"])
    bmi = w / ((h/100)**2)
    cat = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
    bmr = 10*w + 6.25*h - 5*a + (5 if p.get("sex", "male") == "male" else -161)
    tdee = bmr * float(p.get("activity", 1.4))
    target = tdee + {"lose": -500, "gain": 400}.get(p.get("goal", "maintain"), 0)
    target = max(target, 1200)
    prot = w * (1.8 if p.get("goal") != "maintain" else 1.4)
    return dict(bmi=round(bmi, 1), category=cat, bmr=round(bmr), tdee=round(tdee), target=round(target), protein_g=round(prot),
                fat_g=round(target*.27/9), carbs_g=round((target - prot*4 - target*.27)/4), water_l=round(w*0.035, 1),
                ideal_weight=f"{round(18.5*(h/100)**2)}-{round(24.9*(h/100)**2)} kg")

@app.get("/api/summary")
def summary(u=Depends(current_user)):
    today = str(datetime.date.today())
    with db() as c:
        meals = [dict(r) for r in c.execute("SELECT * FROM meals WHERE uid=? AND day=? ORDER BY id", (u["id"], today))]
        wk = {r["day"]: r["k"] for r in c.execute("SELECT day, SUM(calories) k FROM meals WHERE uid=? GROUP BY day ORDER BY day DESC LIMIT 7", (u["id"],))}
    tot = {k: round(sum(m[k] for m in meals), 1) for k in ("calories", "protein", "fat", "carbs", "fiber", "sugar", "sodium")}
    prof = json.loads(u["profile"]); plan = calc_plan(prof) if prof else None
    days = [str(datetime.date.today() - datetime.timedelta(days=i)) for i in range(6, -1, -1)]
    return dict(meals=meals, total=tot, plan=plan, week=[{"day": d[5:], "kcal": round(wk.get(d, 0))} for d in days])

# ---------------- BMI + Diet plan ----------------
class Profile(BaseModel):
    age: int; weight: float; height: float; sex: str = "male"; activity: float = 1.4; goal: str = "maintain"
    diet: str = "vegetarian"; cuisine: str = "Indian"; conditions: str = ""

@app.post("/api/profile")
def save_profile(p: Profile, u=Depends(current_user)):
    with db() as c: c.execute("UPDATE users SET profile=? WHERE id=?", (p.model_dump_json(), u["id"]))
    return calc_plan(p.model_dump())

@app.post("/api/diet-plan")
def diet_plan(p: Profile, u=Depends(current_user)):
    save_profile(p, u); plan = calc_plan(p.model_dump())
    prompt = f"""Create a practical 7-day {p.diet} {p.cuisine} diet plan for: age {p.age}, {p.sex}, {p.weight} kg, {p.height} cm, BMI {plan['bmi']} ({plan['category']}),
goal: {p.goal}, activity factor {p.activity}, health conditions/allergies: {p.conditions or 'none'}.
Daily target: {plan['target']} kcal, protein {plan['protein_g']} g, fat {plan['fat_g']} g, carbs {plan['carbs_g']} g. Use affordable everyday foods with portion sizes.
Return ONLY JSON: {{"summary":str,"days":[{{"day":"Monday","meals":[{{"name":"Breakfast|Mid-morning|Lunch|Evening snack|Dinner","menu":str,"kcal":number}}],"total_kcal":number}}],
"eat_more":[str],"avoid":[str],"habits":[str],"disclaimer":"General guidance, consult a doctor/dietitian for medical conditions."}}"""
    result = gemini([{"text": prompt}]); result["numbers"] = plan
    with db() as c: c.execute("INSERT OR REPLACE INTO plans(uid,plan) VALUES(?,?)", (u["id"], json.dumps(result)))
    return result

@app.get("/api/diet-plan")
def last_plan(u=Depends(current_user)):
    with db() as c: r = c.execute("SELECT plan FROM plans WHERE uid=?", (u["id"],)).fetchone()
    return json.loads(r["plan"]) if r else None

app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
@app.get("/")
def index(): return FileResponse(BASE / "static" / "index.html")
