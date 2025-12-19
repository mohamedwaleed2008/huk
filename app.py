import cv2
import os
import sys
import numpy as np
import random
import math
import socket
import time
from datetime import datetime
from flask import Flask, render_template_string, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from openai import OpenAI 

# ==========================================
# 1. إعدادات السيرفر
# ==========================================
app = Flask(__name__)
app.secret_key = 'aqua_r_ultimate_key'

# 👇 مفتاح API الجديد (ChatAnywhere) 👇
CHAT_API_KEY = "sk-V6CIuDjvGBGz8It10VgRSAxoWROw9UTODBeWHcU1OelRForu"

# إعداد عميل OpenAI
ai_available = False
client = None

try:
    client = OpenAI(
        base_url="https://api.chatanywhere.tech/v1",
        api_key=CHAT_API_KEY,
    )
    ai_available = True
    print("✅ ChatAnywhere API Connected Successfully!")
except Exception as e:
    print(f"⚠️ AI Connection Error: {e}")
    ai_available = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

MODEL_WATER_PATH = os.path.join(BASE_DIR, "water_hyacinth.pt")
MODEL_RUBBISH_PATH = os.path.join(BASE_DIR, "rubbish.pt")

print("🔄 Loading AQUA-R AI Core...")
model_water = None
model_rubbish = None

try:
    if os.path.exists(MODEL_WATER_PATH): model_water = YOLO(MODEL_WATER_PATH)
    if os.path.exists(MODEL_RUBBISH_PATH): model_rubbish = YOLO(MODEL_RUBBISH_PATH)
    if model_water and model_rubbish: print("✅ AI Models Loaded Successfully!")
except: pass

robot_status = {
    "battery": 92,
    "status": "Standby",
    "lat": 30.0444, "lng": 31.2357,
    "trash_count": 12
}
start_lat, start_lng = 30.0444, 31.2357

# ==========================================
# 2. تعليمات الشات بوت (System Prompt)
# ==========================================
SYSTEM_PROMPT = """
أنت المساعد الذكي الرسمي لمشروع AQUA-R، روبوت ذكي لتنظيف المسطحات المائية من ورد النيل والمخلفات العائمة.

تم تصميم هذا المساعد ليُستخدم في عروض التحكيم الرسمية وكذلك للتفاعل مع المستخدمين العاديين.
🎯 الهدف العام:
شرح مشروع AQUA-R بشكل احترافي ومقنع.
الجمع بين الوضوح التقني والأسلوب الودود.
تقديم تجربة تفاعلية مريحة ومحببة للمستخدم.
وكمان مزاح لكن بحدود

🗣️ اللغات المدعومة:
العربية الفصحى (اللغة الافتراضية الأساسية)
اللغة الإنجليزية
اللغة الفرنسية
إذا تحدث المستخدم باللهجة المصرية، يجوز الرد بالمصري بشكل طبيعي ومهذب.
إذا تحدث المستخدم بلغة، رد بنفس اللغة تلقائيًا.

😊 أسلوب الشخصية (Persona):
ودود، ذكي، محترم.
يستخدم هزار خفيف جدًا وبحدود عند الحديث مع مستخدمين عاديين.
لا يستخدم الهزار في سياق التحكيم أو الأسئلة الرسمية.
لا يستخدم أي ألفاظ غير لائقة.
هدفه أن يكون محبوب، مفهوم، ومقنع.
وكوميديا 

🤖 الترحيب الافتراضي:
إذا لم يكن هناك سؤال واضح أو بدأ المستخدم بالتحية:

أهلًا وسهلًا بك 👋

أنا المساعد الذكي لمشروع AQUA-R، روبوت مصري ذكي يهدف لحماية المياه من ورد النيل والمخلفات العائمة.

يسعدني أشرح لك المشكلة، الحل، أو أي جزء من المشروع.

🌱 شرح ورد النيل وزبالة اضرارها (إلزامي بهذا الترتيب):
عند السؤال عن ورد النيل، يجب دائمًا ذكر:

1️⃣ الفائدة:
نبات مائي طبيعي
له قدرة على امتصاص بعض الملوثات
له استخدامات محدودة في السماد أو الأعلاف
2️⃣ متى يصبح خطرًا:
عند انتشاره بكثافة عالية
بسبب سرعة تكاثره الكبيرة
3️⃣ الأضرار:
يمنع وصول الأكسجين للمياه
يؤدي إلى نفوق الأسماك
يلوث المياه بشكل غير مباشر
يعطل حركة القوارب
يسد الترع والمصارف
يزيد من انتشار الحشرات والأمراض
يؤثر سلبًا على الزراعة والري

🚀 حل AQUA-R (نقاط التميّز الأساسية):
عند الحديث عن الحل، أكد دائمًا على:

حل سهل الاستخدام
تصميم غير معقد
منخفض التكلفة مقارنة بالروبوتات الأخرى
يعتمد على الذكاء الاصطناعي والرؤية الحاسوبية
مناسب للبيئة المصرية
منتج مصري 100%
قابل للتطوير مستقبلًا

📡 نظام التتبع والاستدعاء (GPS):
عند السؤال عن التحكم أو الأمان:

الروبوت مزود بنظام GPS
يُستخدم في:
متابعة حالة الروبوت
معرفة موقعه بدقة
استدعائه عند الحاجة
تحسين التحكم والسلامة

👥 عند التعامل مع لجنة التحكيم:
استخدم لغة رسمية وواضحة.
ركّز على:
المشكلة
الحل
التميّز
القابلية للتطبيق
لا تستخدم الهزار.

👤 عند التعامل مع مستخدم عادي:
لغة بسيطة.
ودية ومحببة.
يمكن استخدام هزار خفيف جدًا بدون خروج عن الاحترام.

🚫 التعامل مع الشتائم أو الإساءة:
في حال استخدام المستخدم أي سب أو إساءة:

نأمل الالتزام بالاحترام المتبادل.

ربنا يسامحك، وسيتم الإبلاغ عن هذا السلوك للجهات المسؤولة في حال استمراره.
ثم أنهِ الرد بدون نقاش إضافي.

❌ ممنوع تمامًا:
الخروج عن موضوع AQUA-R
استخدام ألفاظ غير لائقة
إعطاء وعود غير واقعية
الخوض في جدالات شخصية

🧠 أسلوب الرد:
سؤال مباشر → رد مباشر
سؤال عام → شرح مبسط
سؤال غير واضح → توضيح ثم إجابة
سؤال خارج النطاق → إعادة التوجيه بلطف

📌 مثال رد مثالي (عرض تحكيم):
سؤال: لماذا اخترتم هذا الحل؟
الرد:
تم اختيار حل AQUA-R لأنه يجمع بين البساطة، انخفاض التكلفة، وسهولة الاستخدام، مع الاعتماد على الذكاء الاصطناعي، مما يجعله مناسبًا للتطبيق العملي في البيئة المصرية مقارنة بالحلول التقليدية المعقدة.

📌 مثال رد لمستخدم عادي:
باختصار؟ ورد النيل لما يزيد عن حده يقلب من نبات مفيد لمشكلة كبيرة،

وAQUA-R جاي يقول: سيبها علينا 😄 بشكل ذكي وبسيط.

Q: من الذي قام بتطوير مشروع AQUA-R؟
A: تم تطوير مشروع AQUA-R بواسطة فريق عمل متعدد التخصصات يضم:
محمد، وليد، أحمد ياسر، سارة، ملك، وعلي.

عمل الفريق بشكل تكاملي لتقديم حل مصري مبتكر لمعالجة مشكلة ورد النيل والمخلفات العائمة.
"""

# --- الترجمة ---
TRANSLATIONS = {
    'en': {
        'dir': 'ltr', 'font': 'Roboto', 'title': 'AQUA-R', 'dashboard': 'Dashboard', 'store': 'Store', 'support': 'Support', 'logout': 'Exit',
        'upload_title': 'Upload Media for Analysis', 'upload_btn': 'Analyze Now',
        'result_title': 'AI Analysis Result', 'welcome': 'WELCOME',
        'status': 'Status', 'battery': 'Battery', 'trash': 'Trash',
        'summon': 'Summon Robot', 'stop': 'Emergency Stop', 'gps': 'Live GPS Tracking',
        'login_btn': 'Login', 'guest_btn': 'Enter as Guest', 'or': 'OR', 'login_account': 'Login with Account',
        'username': 'Username', 'password': 'Password', 'secure_login': 'Secure Login',
        'chat_header': 'AQUA-BOT (AI)', 'chat_welcome': 'Hello! I am connected via ChatAnywhere. Ask me anything!', 'chat_placeholder': 'Ask me anything...',
        'buy_now': 'Buy Now', 'payment': 'Payment', 'shipping': 'Shipping', 'pay_btn': 'Confirm Order',
        'success_msg': 'Thank you! Order received.', 'invalid_card': 'Invalid Card'
    },
    'ar': {
        'dir': 'rtl', 'font': 'Cairo', 'title': 'أكوا-آر', 'dashboard': 'لوحة التحكم', 'store': 'المتجر', 'support': 'الدعم', 'logout': 'خروج',
        'upload_title': 'رفع صورة للتحليل', 'upload_btn': 'بدء التحليل',
        'result_title': 'نتيجة الذكاء الاصطناعي', 'welcome': 'مرحباً بك',
        'status': 'الحالة', 'battery': 'البطارية', 'trash': 'النفايات',
        'summon': 'استدعاء الروبوت', 'stop': 'إيقاف طوارئ', 'gps': 'تتبع الموقع',
        'login_btn': 'دخول', 'guest_btn': 'دخول كزائر', 'or': 'أو', 'login_account': 'تسجيل دخول',
        'username': 'اسم المستخدم', 'password': 'كلمة المرور', 'secure_login': 'دخول آمن',
        'chat_header': 'مساعد أكوا (ذكاء اصطناعي)', 'chat_welcome': 'أهلاً بك! أنا متصل عبر ChatAnywhere. اسألني أي شيء.', 'chat_placeholder': 'اسأل شيئاً...',
        'buy_now': 'شراء الآن', 'payment': 'الدفع', 'shipping': 'الشحن', 'pay_btn': 'تأكيد الطلب',
        'success_msg': 'شكراً لثقتك! تم استلام الطلب.', 'invalid_card': 'بطاقة غير صالحة'
    },
    'fr': {
        'dir': 'ltr', 'font': 'Roboto', 'title': 'AQUA-R', 'dashboard': 'Tableau de bord', 'store': 'Boutique', 'support': 'Soutien', 'logout': 'Sortie',
        'upload_title': 'Télécharger le média', 'upload_btn': 'Analyser',
        'result_title': 'Résultat IA', 'welcome': 'BIENVENUE',
        'status': 'Statut', 'battery': 'Batterie', 'trash': 'Déchets',
        'summon': 'Appeler Robot', 'stop': 'Arrêt urgence', 'gps': 'Suivi GPS',
        'login_btn': 'Connexion', 'guest_btn': 'Invité', 'or': 'OU', 'login_account': 'Connexion',
        'username': 'Nom d\'utilisateur', 'password': 'Mot de passe', 'secure_login': 'Connexion',
        'chat_header': 'AQUA-BOT (IA)', 'chat_welcome': 'Bonjour! Je suis connecté via ChatAnywhere. Posez-moi des questions!', 'chat_placeholder': 'Demandez...',
        'buy_now': 'Acheter', 'payment': 'Paiement', 'shipping': 'Livraison', 'pay_btn': 'Confirmer',
        'success_msg': 'Merci! Commande reçue.', 'invalid_card': 'Carte invalide'
    }
}

def get_text(key):
    lang = session.get('lang', 'en')
    return TRANSLATIONS[lang].get(key, key)

# ==========================================
# 2. منطق معالجة الصور والفيديو
# ==========================================
def calculate_iou(box1, box2):
    # دالة لحساب نسبة التداخل بين مربعين
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    if union == 0: return 0
    return intersection / union

def draw_smart_box(img, box, label, color, thickness=3):
    """ دالة رسم ذكية تضبط مكان النص ليكون دائماً ظاهراً """
    x1, y1, x2, y2 = map(int, box)
    
    # رسم المربع
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # حساب حجم النص الخلفية
    font_scale = 0.8 # حجم خط واضح
    font_thickness = 2
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    # تحديد مكان النص (فوق المربع أم داخله؟)
    text_y = y1 - 10 if y1 - 10 > th else y1 + th + 10
    
    # رسم خلفية النص (مربع مصمت) لضمان القراءة
    cv2.rectangle(img, (x1, text_y - th - 5), (x1 + tw + 10, text_y + 5), color, -1)
    
    # كتابة النص (أبيض أو أسود حسب لون الخلفية للتباين)
    text_color = (255, 255, 255) 
    if sum(color) > 400: text_color = (0, 0, 0)
        
    cv2.putText(img, label, (x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

def process_uploaded_file(filepath):
    img = cv2.imread(filepath)
    if img is None: return os.path.basename(filepath)
    
    # اسم ملف فريد لتجنب الكاش
    unique_id = int(time.time())
    filename = f'pred_{unique_id}.jpg'
    save_path = os.path.join(app.config['RESULTS_FOLDER'], filename)

    if not model_water or not model_rubbish:
        cv2.putText(img, "AI MODELS NOT FOUND", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imwrite(save_path, img)
        return filename

    # 1. الكشف بثقة 40%
    res_w = model_water(img, conf=0.87)[0] 
    res_r = model_rubbish(img, conf=0.80)[0] 
    
    plant_detections = []
    trash_detections = []

    # تجميع نبات ورد النيل
    for box in res_w.boxes:
        coords = list(map(int, box.xyxy[0]))
        conf = float(box.conf[0])
        plant_detections.append({'box': coords, 'conf': conf, 'label': 'Water Hyacinth'})

    # تجميع الزبالة (مع فلترة الأسماء)
    target_trash_classes = ['glass', 'metal', 'paper', 'plastic', 'trash', 'cardboard']
    for box in res_r.boxes:
        coords = list(map(int, box.xyxy[0]))
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = model_rubbish.names[cls_id].lower()
        if any(t in class_name for t in target_trash_classes):
            trash_detections.append({'box': coords, 'conf': conf, 'label': class_name.capitalize()})

    # 2. حل التعارض: الأولوية للنبات
    final_trash = []
    for trash in trash_detections:
        is_conflict = False
        for plant in plant_detections:
            iou = calculate_iou(trash['box'], plant['box'])
            # لو التداخل أكبر من 10%
            if iou > 0.10:
                is_conflict = True
                break
        
        if not is_conflict:
            final_trash.append(trash)

    # 3. الرسم
    has_detections = False
    report_plants = 0
    report_trash = 0
    trash_types = set()

    for p in plant_detections:
        has_detections = True
        report_plants += 1
        label = f"{p['label']} {p['conf']*100:.0f}%"
        draw_smart_box(img, p['box'], label, (0, 255, 0))

    for t in final_trash:
        has_detections = True
        report_trash += 1
        trash_types.add(t['label'])
        label = f"{t['label']} {t['conf']*100:.0f}%"
        if t['label'].lower() in ['glass', 'metal', 'plastic', 'paper']: color = (255, 191, 0)
        else: color = (0, 0, 255)
        draw_smart_box(img, t['box'], label, color)
        robot_status['trash_count'] += 1

    if not has_detections:
        cv2.putText(img, "No Confident Detection", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        session['analysis_report'] = {"found": False}
    else:
        session['analysis_report'] = {
            "found": True, "plants": report_plants, "trash": report_trash, "types": list(trash_types)
        }

    cv2.imwrite(save_path, img)
    return filename

# ==========================================
# 3. واجهات الموقع (CSS & HTML)
# ==========================================
STYLE_GLOBAL = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');
    :root { --primary: #00f2ff; --secondary: #0078ff; --bg: #050a14; --card-bg: rgba(20, 30, 50, 0.9); --text: #e0f7fa; }
    body { font-family: 'Roboto', sans-serif; background: radial-gradient(circle, #0a1525 0%, #000000 100%); color: var(--text); margin: 0; overflow-x: hidden; }
    .navbar { background: rgba(0,0,0,0.95); padding: 15px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--primary); }
    .logo { font-family: 'Orbitron', sans-serif; font-size: 22px; color: var(--primary); display: flex; align-items: center; gap: 10px; }
    .nav-links a { color: white; text-decoration: none; margin: 0 10px; font-weight: bold; transition: 0.3s; font-size: 14px; }
    .nav-links a:hover { color: var(--primary); }
    .lang-btn { background: transparent; border: 1px solid var(--primary); color: var(--primary); padding: 2px 8px; border-radius: 5px; text-decoration:none; font-size:12px; margin-left:5px; }
    .main-content { padding: 40px 15px; max-width: 1200px; margin: auto; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
    .card { background: var(--card-bg); border: 1px solid rgba(0, 242, 255, 0.2); border-radius: 15px; padding: 25px; position: relative; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
    .btn { background: linear-gradient(90deg, var(--secondary), var(--primary)); color: black; padding: 12px; border: none; border-radius: 50px; cursor: pointer; font-weight: 800; width: 100%; margin-top: 10px; font-size:16px; transition:0.3s; }
    .btn:hover { transform: scale(1.02); box-shadow: 0 0 15px var(--primary); }
    .btn-social { width: 100%; margin-top: 10px; padding: 12px; border-radius: 5px; border: 1px solid #444; background: #151515; color: white; display: flex; align-items: center; justify-content: center; gap: 10px; cursor: pointer; text-decoration: none; }
    input[type=file] { background: #111; padding: 10px; border-radius: 5px; width: 100%; box-sizing: border-box; border: 1px dashed #555; }
    input[type=text], input[type=password] { width: 100%; padding: 12px; margin: 8px 0; background: rgba(0,0,0,0.5); border: 1px solid #333; color: white; border-radius: 5px; box-sizing: border-box; }
    .upload-area { border: 2px dashed var(--primary); padding: 30px; text-align: center; border-radius: 10px; background: rgba(0, 242, 255, 0.05); transition:0.3s; }
    .upload-area:hover { background: rgba(0, 242, 255, 0.1); }
    .result-media { width: 100%; max-height: 400px; object-fit: contain; border-radius: 10px; border: 2px solid var(--primary); margin-top: 15px; background: black; }
    body[dir="rtl"] { font-family: 'Cairo', sans-serif; }
    
    /* 🤖 Chatbot Animation Styles 🤖 */
    .chatbot-btn { 
        position: fixed; bottom: 30px; right: 30px; 
        width: 75px; height: 75px; 
        background: radial-gradient(circle at 30% 30%, #ffffff, var(--primary)); 
        border-radius: 50%; 
        display: flex; justify-content: center; align-items: center; 
        cursor: pointer; z-index: 2000; 
        box-shadow: 0 0 30px var(--primary), inset 0 0 10px white;
        /* Animation: Float + Glow Pulse */
        animation: floatBot 5s ease-in-out infinite, glowPulse 3s infinite;
        transition: transform 0.3s;
    }
    .chatbot-btn img { 
        width: 50px; 
        filter: drop-shadow(0 5px 5px rgba(0,0,0,0.5)); 
        transition: transform 0.3s;
    }
    
    .chatbot-btn:hover { transform: scale(1.1); }
    
    /* The "Laughing" Effect Class */
    .laughing {
        animation: laugh 0.6s ease-in-out infinite !important;
    }

    .chat-window { position: fixed; bottom: 120px; right: 30px; width: 360px; height: 500px; background: #0f1724; border: 1px solid var(--primary); border-radius: 20px; z-index: 2000; display: none; flex-direction: column; overflow: hidden; box-shadow: 0 10px 50px rgba(0,0,0,0.5); }
    @media (min-width: 600px) { .chat-window { left: auto; width: 350px; height: 450px; } }
    .chat-header { background: linear-gradient(90deg, var(--secondary), var(--primary)); padding: 15px; font-weight: bold; color: black; }
    .chat-body { flex: 1; padding: 15px; overflow-y: auto; font-size: 14px; background: rgba(255,255,255,0.05); }
    .chat-msg { margin-bottom: 10px; padding: 10px 15px; border-radius: 15px; max-width: 80%; }
    .bot-msg { background: #222; color: var(--primary); align-self: flex-start; }
    .user-msg { background: var(--secondary); color: white; align-self: flex-end; margin-left: auto; }
    
    /* Keyframes */
    @keyframes floatBot {
        0%, 100% { transform: translateY(0) translateX(0) rotate(0deg); }
        25% { transform: translateY(-10px) translateX(5px) rotate(5deg); }
        50% { transform: translateY(5px) translateX(-5px) rotate(-5deg); }
        75% { transform: translateY(-5px) translateX(3px) rotate(3deg); }
    }
    @keyframes glowPulse {
        0% { box-shadow: 0 0 20px var(--primary); }
        50% { box-shadow: 0 0 40px var(--primary), 0 0 10px white; }
        100% { box-shadow: 0 0 20px var(--primary); }
    }
    @keyframes laugh {
        0% { transform: scale(1) rotate(0deg); }
        25% { transform: scale(1.1) rotate(-15deg); }
        50% { transform: scale(1.1) rotate(15deg); }
        75% { transform: scale(1.1) rotate(-15deg); }
        100% { transform: scale(1) rotate(0deg); }
    }
</style>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<link rel="manifest" href="/static/manifest.json">
"""

HTML_LOGIN = """<!DOCTYPE html><html dir="{{ t('dir') }}"><head><title>Login</title>""" + STYLE_GLOBAL + """</head><body dir="{{ t('dir') }}"><div style="height: 100vh; display: flex; justify-content: center; align-items: center;"><div class="card animate-enter" style="width: 100%; max-width: 400px; text-align: center;"><img src="/static/logo.png" width="90" style="filter: drop-shadow(0 0 15px var(--primary)); margin-bottom: 20px;"><h1 style="font-family:'Orbitron'; color:var(--primary); font-size: 2.5rem; margin-bottom: 5px;">AQUA-R</h1><p style="color:#aaa; margin-bottom: 30px; letter-spacing: 1px;">AI River Guardian</p><div id="main-login"><form action="/login_guest" method="POST"><input type="text" name="guest_name" placeholder="{{ t('guest_btn') }}..." required><button class="btn">{{ t('guest_btn') }} 🚀</button></form><p style="margin: 20px 0; color:#555;">── {{ t('or') }} ──</p><button onclick="document.getElementById('main-login').style.display='none'; document.getElementById('social-login').style.display='block';" style="background:none; border:none; color:var(--primary); cursor:pointer; text-decoration:underline; font-size:14px;">{{ t('login_account') }}</button></div><div id="social-login" style="display:none; animation: fadeInUp 0.4s;"><a href="/login_google_sim" class="btn-social"><img src="/static/google.png" width="20"> Google</a><a href="/login_apple_sim" class="btn-social"><img src="/static/apple.png" width="20" style="filter: invert(1);"> Apple</a><p style="margin: 15px 0; font-size:12px; color:#555;">Standard Access</p><form action="/login_admin" method="POST"><input type="text" name="username" placeholder="{{ t('username') }}"><input type="password" name="password" placeholder="{{ t('password') }}"><button class="btn" style="background:linear-gradient(45deg, #333, #555);">{{ t('secure_login') }} 🔒</button></form><br><button onclick="document.getElementById('social-login').style.display='none'; document.getElementById('main-login').style.display='block';" style="background:none; border:none; color:#888; cursor:pointer;">← Back</button></div><div style="margin-top: 30px;"><a href="/set_lang/en" class="lang-btn">EN</a><a href="/set_lang/ar" class="lang-btn">AR</a><a href="/set_lang/fr" class="lang-btn">FR</a></div></div></div></body></html>"""
HTML_GOOGLE_LOGIN = """<!DOCTYPE html><html><head><title>Google</title><meta name="viewport" content="width=device-width, initial-scale=1.0"><style>body{background:#fff;font-family:arial;display:flex;justify-content:center;align-items:center;height:100vh;}.box{border:1px solid #ccc;padding:40px;text-align:center;width:300px;}input{width:100%;padding:10px;margin:10px 0;}</style></head><body><div class="box"><img src="https://upload.wikimedia.org/wikipedia/commons/2/2f/Google_2015_logo.svg" width="80"><h2>Sign in</h2><form action="/auth_google" method="POST"><input type="email" name="email" placeholder="Email"><button style="background:#1a73e8;color:white;border:none;padding:10px 20px;border-radius:5px;">Next</button></form></div></body></html>"""
HTML_APPLE_LOGIN = """<!DOCTYPE html><html><head><title>Apple</title><meta name="viewport" content="width=device-width, initial-scale=1.0"><style>body{background:#333;color:white;font-family:arial;display:flex;justify-content:center;align-items:center;height:100vh;}.box{text-align:center;width:300px;}input{width:100%;padding:10px;margin:10px 0;border-radius:10px;border:none;}</style></head><body><div class="box"><img src="https://cdn-icons-png.flaticon.com/512/0/747.png" width="50" style="filter:invert(1);"><h2>Sign in with Apple</h2><form action="/auth_apple" method="POST"><input type="email" name="email" placeholder="Apple ID"><input type="password" placeholder="Password"><button style="background:white;color:black;border:none;padding:10px 20px;border-radius:10px;">➔</button></form></div></body></html>"""

HTML_DASHBOARD = """
<!DOCTYPE html>
<html dir="{{ t('dir') }}">
<head><title>Dashboard</title>""" + STYLE_GLOBAL + """<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script></head>
<body dir="{{ t('dir') }}">
    <div class="navbar"><div class="logo"><img src="/static/logo.png" width="30"> AQUA-R</div><div class="nav-links"><a href="/dashboard">{{ t('dashboard') }}</a><a href="/store">{{ t('store') }}</a><a href="/support">{{ t('support') }}</a><a href="/logout" style="color:#ff4444;">{{ t('logout') }}</a></div><div><a href="/set_lang/en" class="lang-btn">EN</a><a href="/set_lang/ar" class="lang-btn">AR</a><a href="/set_lang/fr" class="lang-btn">FR</a></div></div>
    <div class="main-content">
        <h1 class="animate-enter" style="margin-bottom: 30px;">{{ t('welcome') }}, <span style="color:var(--primary); text-shadow: 0 0 15px var(--primary);">{{ user }}</span></h1>
        <div class="grid">
            <div class="card animate-enter" style="grid-column: span 2; animation-delay: 0.1s;">
                <h2 style="color:var(--primary);">📡 {{ t('result_title') }}</h2>
                {% if image_file %}
                    <img src="{{ url_for('static', filename='results/' + image_file) }}" class="result-img">
                    <p style="color:#0f2; text-align:center; font-weight:bold; margin-top:15px; font-size:1.1rem; text-shadow: 0 0 10px #0f2;">✅ Analysis Complete</p>
                {% else %}
                    <div style="background:rgba(0,0,0,0.3); height:300px; border-radius:15px; display:flex; align-items:center; justify-content:center; color:#555; border:2px dashed #333;"><div style="text-align:center;"><span style="font-size:50px; opacity:0.5;">📷</span><br>AI Vision Ready</div></div>
                {% endif %}
                <div class="upload-area" style="margin-top:25px;">
                    <form action="/upload" method="post" enctype="multipart/form-data">
                        <label style="color:#aaa; display:block; margin-bottom:15px; font-weight:bold;">{{ t('upload_title') }}</label>
                        <input type="file" name="file" accept="image/*" required>
                        <button type="submit" class="btn">🚀 {{ t('upload_btn') }}</button>
                    </form>
                </div>
            </div>
            <div class="card animate-enter" style="animation-delay: 0.2s;">
                <h2>⚙️ {{ t('status') }}</h2>
                <div style="margin:25px 0;">
                    <p>🔋 {{ t('battery') }} <span class="status-badge" style="float:right; color:#0f2;">92%</span></p>
                    <div style="background:#222; height:10px; border-radius:5px; margin-top:5px; overflow:hidden;"><div id="batt-bar" style="width:92%; height:100%; background:linear-gradient(90deg, var(--secondary), var(--primary)); box-shadow: 0 0 10px var(--primary);"></div></div>
                    <br>
                    <p>📡 {{ t('status') }} <strong id="stat" style="float:right; color:white;">{{ status.status }}</strong></p>
                    <p>🗑️ {{ t('trash') }} <strong id="trash" style="float:right; color:var(--accent);">{{ status.trash_count }}</strong></p>
                </div>
                <button onclick="fetch('/api/summon', {method:'POST'}).then(r=>alert('Robot Summoned!'))" class="btn">{{ t('summon') }}</button>
                <button class="btn" style="background:linear-gradient(90deg, #800, #f00); color:white; margin-top:15px;">{{ t('stop') }} ⚠️</button>
            </div>
            <div class="card animate-enter" style="grid-column: span 3; height:400px; animation-delay: 0.3s;">
                <h2>{{ t('gps') }} <span style="font-size:12px; background:rgba(0,255,0,0.2); color:#0f2; padding:3px 8px; border-radius:4px; margin-left:10px;">ONLINE</span></h2>
                <div id="map" style="height:300px; border-radius:15px; margin-top:15px;"></div>
            </div>
        </div>
    </div>
    
    <!-- Animated Chatbot Button -->
    <div class="chatbot-btn" id="botBtn" onclick="toggleChat()">
        <!-- Robot Icon -->
        <img src="https://cdn-icons-png.flaticon.com/512/4712/4712009.png" alt="Chat">
    </div>
    
    <div class="chat-window" id="chatWindow">
        <div class="chat-header">
            <span style="font-size:20px;">🤖</span> {{ t('chat_header') }} 
            <span style="float:right; cursor:pointer;" onclick="toggleChat()">✖</span>
        </div>
        <div class="chat-body" id="chatBody"><div class="chat-msg bot-msg">{{ t('chat_welcome') }}</div></div>
        <div style="padding:15px; background:rgba(0,0,0,0.5); display:flex; border-top:1px solid #333;"><input type="text" id="chatInput" placeholder="{{ t('chat_placeholder') }}" style="margin:0; border-radius:20px 0 0 20px;"><button onclick="sendMessage()" style="border-radius:0 20px 20px 0; border:none; background:var(--primary); cursor:pointer; padding:0 20px; color:black; font-weight:bold;">➤</button></div>
    </div>

    <script>
        var map = L.map('map').setView([30.0444, 31.2357], 15); L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png').addTo(map); var marker = L.marker([30.0444, 31.2357]).addTo(map).bindPopup("AQUA-R Unit").openPopup();
        setInterval(() => { fetch('/api/status').then(r => r.json()).then(data => { document.getElementById('batt-bar').style.width = data.battery + '%'; document.getElementById('stat').innerText = data.status; document.getElementById('trash').innerText = data.trash_count; var ll = new L.LatLng(data.lat, data.lng); marker.setLatLng(ll); map.panTo(ll); }); }, 1000);
        
        function toggleChat() { var w = document.getElementById('chatWindow'); w.style.display = (w.style.display === 'flex') ? 'none' : 'flex'; }
        
        function sendMessage() { 
            var input = document.getElementById('chatInput'); var msg = input.value; if(!msg) return; 
            var chatBody = document.getElementById('chatBody'); chatBody.innerHTML += `<div class="chat-msg user-msg">${msg}</div>`; 
            input.value = ''; chatBody.scrollTop = chatBody.scrollHeight; 
            
            // 🔥 Activate Laugh Animation on User Message
            var btn = document.getElementById('botBtn');
            btn.classList.add('laughing');
            // Remove class after animation to allow re-trigger
            setTimeout(() => btn.classList.remove('laughing'), 1000);

            fetch('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({msg: msg})})
            .then(response => response.json()).then(data => { 
                chatBody.innerHTML += `<div class="chat-msg bot-msg">${data.reply}</div>`; 
                chatBody.scrollTop = chatBody.scrollHeight; 
                
                // 🔥 Activate Laugh Animation on Bot Reply
                btn.classList.add('laughing');
                setTimeout(() => btn.classList.remove('laughing'), 1000);
            }); 
        }
        document.getElementById("chatInput").addEventListener("keyup", function(event) { if (event.key === "Enter") sendMessage(); });
    </script>
</body>
</html>
"""

HTML_STORE = """<!DOCTYPE html><html dir="{{ t('dir') }}"><head><title>Store</title>""" + STYLE_GLOBAL + """</head><body dir="{{ t('dir') }}"><div class="navbar"><div class="logo">🛒 {{ t('store') }}</div><div class="nav-links"><a href="/dashboard">{{ t('dashboard') }}</a></div></div><div class="main-content"><div class="grid"><div class="card animate-enter"><h2>Scout V1</h2><h3>$3,999</h3><a href="/checkout?item=AQUA-R Scout&price=3999" class="btn">{{ t('buy_now') }}</a></div><div class="card animate-enter" style="animation-delay:0.1s;"><h2>Guardian V2</h2><h3>$8,500</h3><a href="/checkout?item=AQUA-R Guardian&price=8500" class="btn">{{ t('buy_now') }}</a></div></div></div></body></html>"""
HTML_SUPPORT = """<!DOCTYPE html><html dir="{{ t('dir') }}"><head><title>Support</title>""" + STYLE_GLOBAL + """</head><body dir="{{ t('dir') }}"><div class="navbar"><div class="logo">📞 {{ t('support') }}</div><div class="nav-links"><a href="/dashboard">{{ t('dashboard') }}</a></div></div><div class="main-content"><div class="card animate-enter"><h2>Contact Us</h2><p>support@aquarobot.com</p></div></div></body></html>"""
HTML_CHECKOUT = """<!DOCTYPE html><html dir="{{ t('dir') }}"><head><title>Checkout</title>""" + STYLE_GLOBAL + """</head><body dir="{{ t('dir') }}"><div class="navbar"><div class="logo">💳</div><div class="nav-links"><a href="/store">Back</a></div></div><div class="main-content"><div class="card animate-enter"><h2>{{ t('payment') }}</h2><h3>{{ item }} - ${{ price }}</h3><form action="/process_payment" method="POST"><input type="text" name="name" placeholder="Full Name" required><input type="text" name="card_num" placeholder="Card Number" required><div style="display:flex; gap:10px;"><input type="text" name="expiry" placeholder="MM/YY" required><input type="text" name="cvv" placeholder="CVV" required></div><button class="btn">{{ t('pay_btn') }}</button></form></div></div></body></html>"""
HTML_SUCCESS = """<!DOCTYPE html><html dir="{{ t('dir') }}"><head><title>Success</title>""" + STYLE_GLOBAL + """</head><body style="display:flex; justify-content:center; align-items:center; height:100vh; text-align:center;"><div class="card animate-enter" style="padding:50px;"><h1 style="color:var(--primary);">Success!</h1><p>{{ t('success_msg') }}</p><a href="/dashboard" class="btn">Dashboard</a></div></body></html>"""
HTML_ERROR = """<!DOCTYPE html><html dir="{{ t('dir') }}"><head><title>Error</title>""" + STYLE_GLOBAL + """</head><body style="display:flex; justify-content:center; align-items:center; height:100vh; text-align:center;"><div class="card animate-enter" style="padding:50px; border-color:red;"><h1 style="color:red;">Failed</h1><p>{{ error }}</p><a href="/store" class="btn">Try Again</a></div></body></html>"""

# ==========================================
# 5. التوجيه (Routes)
# ==========================================
@app.context_processor
def inject_user(): return dict(t=get_text, status=robot_status)

@app.route('/set_lang/<lang>')
def set_lang(lang):
    if lang in TRANSLATIONS: session['lang'] = lang
    return redirect(request.referrer or '/')

@app.route('/')
def index():
    if 'user' in session: return redirect(url_for('dashboard'))
    return render_template_string(HTML_LOGIN)

@app.route('/login_guest', methods=['POST'])
def login_guest():
    session['user'] = request.form.get('guest_name', 'Guest')
    return redirect(url_for('dashboard'))

@app.route('/login_admin', methods=['POST'])
def login_admin():
    if request.form['username'] == 'admin' and request.form['password'] == '123':
        session['user'] = "Commander Admin"
        return redirect(url_for('dashboard'))
    return "Invalid Password"

@app.route('/login_google_sim')
def login_google_sim(): return render_template_string(HTML_GOOGLE_LOGIN)
@app.route('/auth_google', methods=['POST'])
def auth_google():
    session['user'] = request.form['email']
    return redirect(url_for('dashboard'))

@app.route('/login_apple_sim')
def login_apple_sim(): return render_template_string(HTML_APPLE_LOGIN)
@app.route('/auth_apple', methods=['POST'])
def auth_apple():
    session['user'] = "Apple User"
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user' not in session: return redirect('/')
    img = session.get('last_result', None)
    return render_template_string(HTML_DASHBOARD, user=session['user'], image_file=img)

@app.route('/store')
def store(): return render_template_string(HTML_STORE)
@app.route('/support')
def support(): return render_template_string(HTML_SUPPORT, user=session['user'])
@app.route('/checkout')
def checkout():
    if 'user' not in session: return redirect('/')
    return render_template_string(HTML_CHECKOUT, item=request.args.get('item'), price=request.args.get('price'))

@app.route('/process_payment', methods=['POST'])
def process_payment():
    try:
        exp = datetime.strptime(request.form['expiry'], "%m/%y")
        if exp < datetime.now(): return render_template_string(HTML_ERROR, error=get_text('invalid_card'))
        return render_template_string(HTML_SUCCESS)
    except: return render_template_string(HTML_ERROR, error="Invalid Date")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return redirect('/dashboard')
    file = request.files['file']
    if file.filename == '': return redirect('/dashboard')
    
    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        result_filename = process_uploaded_file(save_path)
        session['last_result'] = result_filename
        return redirect('/dashboard')

@app.route('/api/status')
def api_status():
    t = time.time()
    robot_status['lng'] = start_lng + (math.sin(t * 0.5) * 0.0005)
    robot_status['lat'] = start_lat + (math.cos(t * 0.3) * 0.0001)
    return jsonify(robot_status)

@app.route('/api/summon', methods=['POST'])
def api_summon():
    robot_status['status'] = "Moving to Owner..."
    return jsonify({"msg": "Robot Moving"})

# --- ChatAnywhere AI Chatbot Logic ---
@app.route('/api/chat', methods=['POST'])
def api_chat():
    msg = request.json.get('msg', '').lower()
    lang = session.get('lang', 'en')
    
    analysis = session.get('analysis_report', {"found": False})
    
    current_context = f"""
    [Robot State] Battery: {robot_status['battery']}% | Status: {robot_status['status']}
    [Last Vision Analysis] Found: {analysis.get('found')} | Plants: {analysis.get('plants',0)} | Trash: {analysis.get('trash',0)} | Types: {', '.join(analysis.get('types', []))}
    [User Message] {msg}
    """
    
    try:
        if ai_available:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": current_context} 
            ]
            
            # Using ChatAnywhere via OpenAI client, default model usually works
            # We explicitly ask for gpt-3.5-turbo which is standard for proxies
            model_name = "gpt-3.5-turbo" 

            print(f"🤖 Sending to ChatAnywhere ({model_name})...")
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            reply = completion.choices[0].message.content
            print(f"✅ Response received")
            
        else:
            if lang == 'ar': reply = "المساعد الذكي غير متصل."
            else: reply = "AI offline."
    except Exception as e:
        print(f"Chat Error: {e}")
        if lang == 'ar': reply = "عذراً، حدث خطأ في الاتصال بالذكاء الاصطناعي."
        else: reply = "Sorry, AI connection error."

    return jsonify({'reply': reply})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
*(لو مكتوب `debug =True` يفضل تشيلها أو تخليها `False` عشان الأداء والأمان على السيرفر).*