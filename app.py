import tensorflow as tf
import numpy as np
import cv2
from flask import Flask, render_template,request, redirect, jsonify, url_for, session,flash
import os
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import json
from time import time
from collections import defaultdict
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
health_tips=json.load(open(os.getcwd()+'/health_tips.json'))


app = Flask(__name__)
app.secret_key='your_secret_key'

upload_folder = os.path.join('static', 'uploads')
 
app.config['UPLOAD'] = upload_folder

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
DATABASE = os.path.join(PROJECT_ROOT, 'instance', 'database.db')
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///'+DATABASE
db = SQLAlchemy(app)


class MedicalQABot:
    def __init__(self):
                
        self.knowledge_base = {
            'Actinic keratoses and intraepithelial carcinomae': """
                Actinic keratoses (AK) are precancerous skin growths caused by sun damage.
                Symptoms: Rough, scaly patches on sun-exposed areas, may be pink, red, or brown.
                Prevention: 
                - Use broad-spectrum sunscreen (SPF 30+)
                - Wear protective clothing and hats
                - Avoid peak sun hours (10am-4pm)
                - Regular skin examinations
                Treatment options:
                - Topical medications (5-fluorouracil, imiquimod)
                - Cryotherapy (freezing)
                - Photodynamic therapy
                - Surgical removal if necessary
                Complications if untreated: May develop into squamous cell carcinoma.
            """,
            
            'Basal cell carcinoma': """
                Basal cell carcinoma (BCC) is the most common type of skin cancer.
                Symptoms: 
                - Pearly, waxy bumps
                - Flat, flesh-colored or brown scar-like lesions
                - Bleeding or scabbing sores that heal and return
                Prevention:
                - Daily sun protection
                - Regular skin checks
                - Avoid tanning beds
                Treatment options:
                - Surgical excision
                - Mohs surgery
                - Radiation therapy
                - Topical treatments for superficial BCC
                Risk factors: Sun exposure, fair skin, radiation therapy, age over 50.
            """,
            
            'Benign keratosis-like lesions': """
                Benign keratosis includes seborrheic keratoses and other non-cancerous growths.
                Characteristics:
                - Waxy, scaly, slightly raised growths
                - Light brown to black in color
                - May appear "stuck on" the skin
                Management:
                - Usually no treatment needed
                - Can be removed for cosmetic reasons
                - Monitor for changes
                When to seek care:
                - Rapid growth or change in appearance
                - Bleeding or itching
                - Inflammation or irritation
            """,
            
            'Dermatofibroma': """
                Dermatofibroma is a common benign skin tumor.
                Features:
                - Small, firm, raised growths
                - Brown to reddish color
                - Most common on legs
                - Usually harmless
                Characteristics:
                - Dimples when pinched
                - May be tender or itchy
                Treatment:
                - Usually unnecessary
                - Surgical removal if problematic
                - Monitoring for changes
                Prevention:
                - Protect from injury
                - Avoid picking or trauma
            """,
            
            'Melanoma': """
                Melanoma is the most serious type of skin cancer.
                ABCDE warning signs:
                - Asymmetry
                - Border irregularity
                - Color variation
                - Diameter larger than 6mm
                - Evolving size/shape/color
                Risk factors:
                - UV exposure
                - Fair skin
                - Family history
                - Multiple moles
                Prevention:
                - Regular skin checks
                - Sun protection
                - Early detection crucial
                Treatment:
                - Surgery
                - Immunotherapy
                - Targeted therapy
                Requires immediate medical attention if suspected.
            """,
            
            'Melanocytic nevi': """
                Melanocytic nevi (moles) are common skin growths.
                Types:
                - Congenital (present at birth)
                - Acquired (develop over time)
                - Atypical (unusual appearance)
                Monitoring:
                - Regular self-examination
                - Document changes
                - Professional evaluation
                Warning signs:
                - Changes in size/shape
                - Color changes
                - Irregular borders
                Prevention:
                - Sun protection
                - Regular monitoring
                - Professional skin checks
            """,
            
            'Pyogenic granulomas and hemorrhage': """
                Pyogenic granulomas are rapidly growing skin growths.
                Characteristics:
                - Small, round growths
                - Bright red color
                - Bleeds easily
                - Often appears after injury
                Treatment options:
                - Surgical removal
                - Laser therapy
                - Electrocauterization
                Care instructions:
                - Keep area clean
                - Avoid trauma
                - Protect from friction
                When to seek care:
                - Excessive bleeding
                - Rapid growth
                - Signs of infection
            """
        }

    def get_response(self, question, disease):
        context = self.knowledge_base.get(disease, "")
        if not context:
            return "I don't have specific information about this condition. Please consult a healthcare professional."
        
        try:
            # Simple keyword-based response
            question = question.lower()
            
            # Define common question patterns
            if any(word in question for word in ['symptom', 'sign']):
                if 'Symptoms:' in context:
                    return self._extract_section(context, 'Symptoms:')
            elif any(word in question for word in ['treat', 'cure', 'therapy']):
                if 'Treatment' in context:
                    return self._extract_section(context, 'Treatment')
            elif any(word in question for word in ['prevent', 'avoid']):
                if 'Prevention:' in context:
                    return self._extract_section(context, 'Prevention:')
            elif any(word in question for word in ['risk', 'cause']):
                if 'Risk factors:' in context:
                    return self._extract_section(context, 'Risk factors:')
            else:
                # Return a general response
                return f"This condition is {disease}. Please ask specific questions about symptoms, treatment, prevention, or risk factors."
            
            return "I apologize, but I couldn't find specific information about that. Please try asking about symptoms, treatment, prevention, or risk factors."
            
        except Exception as e:
            print(f"Error in QA system: {str(e)}")
            return "I apologize, but I'm having trouble processing your question. Please consult a healthcare professional."

    def _extract_section(self, text, section_marker):
        """Helper method to extract relevant section from the text."""
        try:
            start = text.index(section_marker)
            next_section = float('inf')
            for marker in ['Symptoms:', 'Treatment', 'Prevention:', 'Risk factors:', 'Characteristics:', 'Features:']:
                if marker != section_marker:
                    try:
                        pos = text.index(marker, start + len(section_marker))
                        next_section = min(next_section, pos)
                    except ValueError:
                        continue
            
            if next_section == float('inf'):
                section_text = text[start:].split('\n\n')[0]
            else:
                section_text = text[start:next_section]
            
            return section_text.strip() + "\n(Note: Please consult healthcare professionals for medical advice)"
        except ValueError:
            return "Information not found in the knowledge base."

# Initialize the chatbot (add this after your model initialization)
chatbot = MedicalQABot()



class RateLimit:
    def __init__(self):
        self.requests = defaultdict(list)
        self.limit = 5  # requests per minute
        self.time_window = 60  # seconds

    def is_allowed(self, user_id):
        current_time = time()
        user_requests = self.requests[user_id]
        
        # Remove old requests
        user_requests = [req_time for req_time in user_requests 
                        if current_time - req_time < self.time_window]
        self.requests[user_id] = user_requests
        
        if len(user_requests) >= self.limit:
            return False
            
        user_requests.append(current_time)
        return True

# Initialize rate limiter (add this after chatbot initialization)
rate_limiter = RateLimit()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self,email,password,name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    
    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8'))

with app.app_context():
    db.create_all()

from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K


def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        focal_loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(focal_loss, axis=-1)
    return loss


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


model = tf.keras.models.load_model(
    "final_model5.keras",
    custom_objects={
        'focal_loss': focal_loss(alpha=0.25, gamma=2.0),
        'recall_m': recall_m,
        'precision_m': precision_m,
        'f1_m': f1_m
    }
)

# model = tf.keras.models.load_model(os.getcwd()+'\\final_model5.keras')


label_mapping = {
    0: 'Actinic keratoses and intraepithelial carcinomae',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic nevi',
    6: 'Pyogenic granulomas and hemorrhage'
}

from PIL import Image
import numpy as np


def predict_skin_disease(image_path):
    print(image_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (28, 28)) 
    img = img / 255.0
    img=img.reshape((28, 28, 3))
    pred = model.predict(np.array([img]))[0]
    predicted_class_index = np.argmax(pred)
    print('index:',predicted_class_index,' ',image_path)
    predicted_class_name = label_mapping[predicted_class_index]
    return predicted_class_name


@app.route('/',methods=["GET","POST"])
def index():
    if session.get("email"):
        user = User.query.filter_by(email=session['email']).first()
        return render_template('index.html',user=user)

    return render_template('index.html')

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if session.get("email"):
        user = User.query.filter_by(email=session['email']).first()
        if request.method == 'POST':
            file = request.files["skin_photo"]
            filename = secure_filename(file.filename)
            upload_dir = os.path.join(app.config['UPLOAD'], filename)
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(upload_dir), exist_ok=True)
            
            file.save(upload_dir)
            result = predict_skin_disease(upload_dir)
            session["disease"] = result
            
            # If it's an AJAX request, return JSON
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'img': upload_dir,
                    'disease': result
                })
            
            # For regular requests, return the full template
            return render_template('predict.html', 
                                img=upload_dir, 
                                disease=result, 
                                user=user)

        return render_template('predict.html', user=user)
    flash('Please Login First')
    return render_template('login.html')


@app.route('/register',methods=["GET","POST"])
def register():
    if request.method=="POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name,email=email,password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
       
    
    return render_template("register.html")

@app.route('/login',methods=["GET","POST"])
def login():
    if request.method=="POST":
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/')
        else:
            return render_template('login.html',error='Invalid user')
    
    return render_template("login.html")


@app.route('/logout')
def logout():
    session.pop('email',None)
    session.pop('disease',None)
    return redirect("/")

@app.route('/Health_Tips')
def Health_Tips():
    if session.get("email") and session.get("disease"):
       user = User.query.filter_by(email=session['email']).first()
       disease=session["disease"]
       tips=health_tips[disease]       
       return render_template("Health_Tips.html",disease=disease,user=user,health_tips_disease=tips,len=len(tips))
    else:
       return redirect("/predict")
@app.route('/chat', methods=['POST'])
def chat():
    if not session.get("email"):
        return jsonify({'response': 'Please login first'})
    
    # Check rate limit
    if not rate_limiter.is_allowed(session['email']):
        return jsonify({'response': 'Please wait a moment before sending another message'}), 429
    
    try:
        data = request.get_json()
        user_message = data.get('message')
        disease = session.get("disease")  # Get disease from session
        
        if not disease:
            return jsonify({'response': 'Please get a disease analysis first'}), 400
        
        # Get response from chatbot
        response = chatbot.get_response(user_message, disease)
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"Error in chat route: {str(e)}")
        return jsonify({
            'response': 'An error occurred while processing your request. Please try again.'
        }), 500

if __name__== "__main__":
    app.run(debug=True)

