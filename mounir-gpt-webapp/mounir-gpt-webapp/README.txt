MOUNIR GPT-5.2 — Web App (FastAPI + Streaming) — Single server demo

تشغيل:
1) ادخل فولدر backend
2) انسخ .env.example ل .env وحط OPENAI_API_KEY بتاعك
3) نفّذ:
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8000

بعدها افتح:
http://localhost:8000

ملاحظات:
- الواجهة هنا HTML بسيطة عشان تقدر "تشوفها" بسرعة من غير Next.js.
- Streaming شغال عبر SSE events.
- حفظ الرسائل في SQLite داخل backend/chat.db
