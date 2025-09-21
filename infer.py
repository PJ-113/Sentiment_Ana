from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np

MODEL_PATH = "./xlmr-finetuned-course-sentiment"
id2label = {0:"negative",1:"neutral",2:"positive"}

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).eval()

def predict(text):
    x = tok(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        p = torch.softmax(model(**x).logits, dim=-1).numpy()[0]
    k = int(np.argmax(p))
    return {
        "text": text,
        "label": id2label[k],
        "probs": {id2label[i]: float(p[i]) for i in range(3)}
    }

# ตัวอย่างข้อความรีวิว
examples = [
    "The professor was very clear and supportive.",
    "Assignments were okay but nothing special.",
    "Exams were too hard and grading was unfair.",
    "อาจารย์อธิบายชัดเจน เข้าใจง่าย",
    "งานเยอะเกินไป คะแนนเข้มมาก",
    "เฉย ๆ ไม่ดีไม่แย่",
    "อาจารย์สอนดีแต่งานเยอะแล้วก็ข้อสอบยาก",
    "เนื้อหาชัดเจน มีโครงสร้างดี เหมาะสำหรับการ ฝึกทำโจทย์"
]

for ex in examples:
    result = predict(ex)
    print(f"\nReview: {result['text']}")
    print(f"Predicted sentiment: {result['label']}")
    print(f"Probabilities: {result['probs']}")


