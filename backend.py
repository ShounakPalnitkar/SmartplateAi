from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
from werkzeug.security import generate_password_hash, check_password_hash
import logging
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
import base64

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['JWT_SECRET_KEY'] = 'jwt-secret-key-here-change-in-production'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

# Initialize extensions
CORS(app)
jwt = JWTManager(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock AI Models (In production, you'd use actual trained models)
class NutritionAI:
    def __init__(self):
        self.portion_model = self._load_mock_model('portion_estimation')
        self.recommendation_model = self._load_mock_model('meal_recommendation')
        self.forecasting_model = self._load_mock_model('grocery_forecasting')
        self.deficiency_model = self._load_mock_model('deficiency_detection')
        self.optimization_model = self._load_mock_model('menu_optimization')
    
    def _load_mock_model(self, model_type):
        # In production, load actual trained models
        return f"mock_{model_type}_model"
    
    def estimate_portions(self, food_image_data):
        """Mock portion estimation using regression algorithms"""
        # In production, this would use computer vision and regression models
        return {
            'calories': np.random.randint(200, 800),
            'protein': round(np.random.uniform(5, 40), 1),
            'carbs': round(np.random.uniform(20, 100), 1),
            'fats': round(np.random.uniform(5, 30), 1)
        }
    
    def recommend_meals(self, user_profile, preferences, budget):
        """Generate personalized meal recommendations"""
        meals = self._load_sample_meals()
        
        # Filter based on preferences
        filtered_meals = []
        for meal in meals:
            if preferences.get('vegetarian', False) and not meal.get('vegetarian', True):
                continue
            if preferences.get('vegan', False) and not meal.get('vegan', False):
                continue
            if preferences.get('gluten_free', False) and not meal.get('gluten_free', True):
                continue
            if any(allergy in meal.get('ingredients', '') for allergy in preferences.get('allergies', [])):
                continue
            
            filtered_meals.append(meal)
        
        # Sort by nutritional score and budget
        recommended = sorted(filtered_meals, 
                           key=lambda x: x.get('nutrition_score', 0), 
                           reverse=True)[:7]
        
        return recommended
    
    def forecast_grocery_costs(self, meal_plan, location):
        """Predict grocery costs using time series forecasting"""
        # Mock implementation - in production use ARIMA, Prophet, or LSTM
        base_cost = sum(meal.get('estimated_cost', 0) for meal in meal_plan)
        
        # Add location-based adjustments
        location_multipliers = {
            'urban': 1.2,
            'suburban': 1.0,
            'rural': 0.9
        }
        
        adjusted_cost = base_cost * location_multipliers.get(location, 1.0)
        return round(adjusted_cost, 2)
    
    def detect_deficiencies(self, user_data, meal_history):
        """Identify potential nutritional deficiencies"""
        deficiencies = []
        
        # Mock deficiency detection logic
        if user_data.get('age', 30) > 50:
            deficiencies.append({
                'nutrient': 'Vitamin D',
                'risk_level': 'medium',
                'recommendation': 'Consider vitamin D supplements or increase fatty fish consumption'
            })
        
        if user_data.get('gender') == 'female' and user_data.get('age', 30) < 50:
            deficiencies.append({
                'nutrient': 'Iron',
                'risk_level': 'high',
                'recommendation': 'Increase iron-rich foods like spinach, lentils, and red meat'
            })
        
        return deficiencies
    
    def optimize_menu(self, available_ingredients, budget, nutritional_goals):
        """Optimize menu for nutritional value and cost"""
        # Mock optimization using genetic algorithms
        return {
            'optimized_meals': self._load_sample_meals()[:5],
            'total_cost': budget * 0.8,
            'nutrition_score': 85,
            'cost_savings': budget * 0.2
        }
    
    def _load_sample_meals(self):
        """Load sample meal data"""
        return [
            {
                'name': 'Avocado Toast with Eggs',
                'calories': 320,
                'protein': 15,
                'carbs': 30,
                'fats': 18,
                'estimated_cost': 2.50,
                'vegetarian': True,
                'vegan': False,
                'gluten_free': True,
                'nutrition_score': 85,
                'ingredients': ['avocado', 'eggs', 'bread', 'lemon']
            },
            {
                'name': 'Quinoa Buddha Bowl',
                'calories': 450,
                'protein': 18,
                'carbs': 55,
                'fats': 20,
                'estimated_cost': 3.50,
                'vegetarian': True,
                'vegan': True,
                'gluten_free': True,
                'nutrition_score': 90,
                'ingredients': ['quinoa', 'chickpeas', 'vegetables', 'tahini']
            },
            # Add more sample meals...
        ]

# Initialize AI models
nutrition_ai = NutritionAI()

# User Management
class UserManager:
    def __init__(self):
        self.users_file = 'users.json'
        self._load_users()
    
    def _load_users(self):
        try:
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
        except FileNotFoundError:
            self.users = {}
    
    def _save_users(self):
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def create_user(self, username, email, password):
        if username in self.users:
            return False, "Username already exists"
        
        self.users[username] = {
            'email': email,
            'password_hash': generate_password_hash(password),
            'created_at': datetime.now().isoformat(),
            'profile': {},
            'meal_history': [],
            'preferences': {}
        }
        self._save_users()
        return True, "User created successfully"
    
    def authenticate_user(self, username, password):
        user = self.users.get(username)
        if user and check_password_hash(user['password_hash'], password):
            return True, user
        return False, "Invalid credentials"
    
    def update_profile(self, username, profile_data):
        if username in self.users:
            self.users[username]['profile'] = profile_data
            self._save_users()
            return True, "Profile updated"
        return False, "User not found"

user_manager = UserManager()

# Routes
@app.route('/')
def home():
    return jsonify({"message": "AI-Powered Smart Plate API", "status": "running"})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    success, message = user_manager.create_user(username, email, password)
    
    if success:
        return jsonify({"message": message}), 201
    else:
        return jsonify({"error": message}), 400

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    success, result = user_manager.authenticate_user(username, password)
    
    if success:
        access_token = create_access_token(identity=username)
        return jsonify({
            "access_token": access_token,
            "user": {
                "username": username,
                "email": result['email']
            }
        }), 200
    else:
        return jsonify({"error": result}), 401

@app.route('/api/assessment', methods=['POST'])
@jwt_required()
def process_assessment():
    current_user = get_jwt_identity()
    data = request.get_json()
    
    try:
        # Process assessment data
        user_profile = {
            'age': data.get('age'),
            'gender': data.get('gender'),
            'height': data.get('height'),
            'weight': data.get('weight'),
            'activity_level': data.get('activity'),
            'goal': data.get('goal'),
            'budget': data.get('budget')
        }
        
        preferences = {
            'vegetarian': data.get('vegetarian', False),
            'vegan': data.get('vegan', False),
            'gluten_free': data.get('glutenFree', False),
            'dairy_free': data.get('dairyFree', False),
            'allergies': [a.strip() for a in data.get('allergies', '').split(',') if a.strip()]
        }
        
        # Generate meal recommendations
        recommended_meals = nutrition_ai.recommend_meals(user_profile, preferences, user_profile['budget'])
        
        # Detect potential deficiencies
        deficiencies = nutrition_ai.detect_deficiencies(user_profile, [])
        
        # Calculate nutritional goals
        nutritional_goals = calculate_nutritional_goals(user_profile)
        
        # Update user profile
        user_manager.update_profile(current_user, {
            **user_profile,
            'preferences': preferences,
            'nutritional_goals': nutritional_goals
        })
        
        response = {
            'meal_plan': recommended_meals,
            'nutritional_goals': nutritional_goals,
            'deficiencies': deficiencies,
            'weekly_cost_estimate': sum(meal.get('estimated_cost', 0) for meal in recommended_meals),
            'nutrition_score': calculate_overall_nutrition_score(recommended_meals)
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Assessment processing error: {str(e)}")
        return jsonify({"error": "Failed to process assessment"}), 500

@app.route('/api/portion-estimation', methods=['POST'])
@jwt_required()
def estimate_portion():
    try:
        # In production, this would process image data
        image_data = request.get_json().get('image_data')
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Analyze food image and estimate portions
        nutrition_info = nutrition_ai.estimate_portions(image_data)
        
        return jsonify({
            "nutrition_info": nutrition_info,
            "confidence_score": round(np.random.uniform(0.7, 0.95), 2)
        }), 200
        
    except Exception as e:
        logger.error(f"Portion estimation error: {str(e)}")
        return jsonify({"error": "Failed to estimate portions"}), 500

@app.route('/api/grocery-forecast', methods=['POST'])
@jwt_required()
def forecast_grocery():
    current_user = get_jwt_identity()
    data = request.get_json()
    
    try:
        meal_plan = data.get('meal_plan', [])
        location = data.get('location', 'suburban')
        
        forecast = nutrition_ai.forecast_grocery_costs(meal_plan, location)
        
        return jsonify({
            "estimated_cost": forecast,
            "forecast_confidence": "high",
            "cost_saving_tips": generate_cost_saving_tips(meal_plan)
        }), 200
        
    except Exception as e:
        logger.error(f"Grocery forecast error: {str(e)}")
        return jsonify({"error": "Failed to generate grocery forecast"}), 500

@app.route('/api/generate-report', methods=['POST'])
@jwt_required()
def generate_report():
    current_user = get_jwt_identity()
    data = request.get_json()
    
    try:
        # Generate PDF report
        pdf_buffer = generate_pdf_report(data)
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f"nutrition_report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        return jsonify({"error": "Failed to generate report"}), 500

@app.route('/api/recipes', methods=['GET'])
def get_recipes():
    try:
        # Get query parameters for filtering
        dietary_preferences = request.args.get('dietary', '')
        max_calories = request.args.get('max_calories', type=int)
        ingredients = request.args.get('ingredients', '')
        
        recipes = nutrition_ai._load_sample_meals()
        
        # Apply filters
        if dietary_preferences:
            preferences = dietary_preferences.split(',')
            recipes = [r for r in recipes if any(r.get(p, False) for p in preferences)]
        
        if max_calories:
            recipes = [r for r in recipes if r.get('calories', 0) <= max_calories]
        
        return jsonify({"recipes": recipes}), 200
        
    except Exception as e:
        logger.error(f"Recipe fetch error: {str(e)}")
        return jsonify({"error": "Failed to fetch recipes"}), 500

@app.route('/api/chatbot', methods=['POST'])
@jwt_required()
def chatbot():
    data = request.get_json()
    message = data.get('message', '')
    
    try:
        # Simple rule-based chatbot for nutrition advice
        response = process_chatbot_message(message)
        
        return jsonify({
            "response": response,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Chatbot error: {str(e)}")
        return jsonify({"error": "Chatbot service unavailable"}), 500

# Utility Functions
def calculate_nutritional_goals(user_profile):
    """Calculate personalized nutritional goals based on user profile"""
    age = user_profile.get('age', 30)
    weight = user_profile.get('weight', 70)
    height = user_profile.get('height', 170)
    activity_level = user_profile.get('activity_level', 'moderate')
    goal = user_profile.get('goal', 'maintenance')
    
    # Calculate BMR (Basal Metabolic Rate)
    if user_profile.get('gender') == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
    # Adjust for activity level
    activity_multipliers = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'extreme': 1.9
    }
    
    tdee = bmr * activity_multipliers.get(activity_level, 1.55)
    
    # Adjust for goals
    if goal == 'weight_loss':
        tdee *= 0.8
    elif goal == 'weight_gain':
        tdee *= 1.2
    elif goal == 'muscle_gain':
        tdee *= 1.1
    
    calories = round(tdee)
    protein = round(weight * 1.6)  # 1.6g per kg for active individuals
    carbs = round((calories * 0.5) / 4)  # 50% of calories from carbs
    fats = round((calories * 0.3) / 9)   # 30% of calories from fats
    
    return {
        'calories': calories,
        'protein': protein,
        'carbs': carbs,
        'fats': fats,
        'fiber': round(weight * 0.4),  # 0.4g per kg
        'water_ml': weight * 30  # 30ml per kg
    }

def calculate_overall_nutrition_score(meals):
    """Calculate overall nutrition score for a meal plan"""
    if not meals:
        return 0
    
    total_score = sum(meal.get('nutrition_score', 0) for meal in meals)
    return round(total_score / len(meals))

def generate_cost_saving_tips(meal_plan):
    """Generate personalized cost-saving tips"""
    tips = [
        "Buy seasonal vegetables to save 15-20%",
        "Purchase grains in bulk for long-term savings",
        "Consider frozen fruits for smoothies",
        "Plan meals around weekly grocery sales",
        "Cook in batches to reduce energy costs"
    ]
    
    # Add specific tips based on meal plan
    if any('avocado' in str(meal.get('ingredients', [])).lower() for meal in meal_plan):
        tips.append("Buy avocados in bulk when on sale and freeze for later use")
    
    return tips

def process_chatbot_message(message):
    """Process chatbot messages and return responses"""
    message_lower = message.lower()
    
    if 'calorie' in message_lower:
        return "Calorie needs vary based on age, weight, height, and activity level. I can help you calculate your specific needs through the assessment."
    elif 'protein' in message_lower:
        return "Most adults need about 0.8-1.2 grams of protein per kg of body weight. Active individuals may need more for muscle repair and growth."
    elif 'recipe' in message_lower or 'meal' in message_lower:
        return "I can suggest recipes based on your dietary preferences and nutritional goals. Check out the recipes section or complete an assessment for personalized recommendations."
    elif 'budget' in message_lower or 'cost' in message_lower:
        return "Eating healthy on a budget is possible! Focus on seasonal produce, buy in bulk, and plan your meals ahead to reduce food waste."
    elif 'deficiency' in message_lower or 'vitamin' in message_lower:
        return "Common deficiencies include Vitamin D, Iron, and B12. Our assessment can help identify potential risks based on your diet and lifestyle."
    else:
        return "I'm here to help with nutrition advice, meal planning, and healthy eating tips. You can ask me about calories, protein, recipes, or budget-friendly eating."

def generate_pdf_report(data):
    """Generate PDF nutrition report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1  # Center
    )
    story.append(Paragraph("AI-Powered Smart Plate Nutrition Report", title_style))
    story.append(Spacer(1, 12))
    
    # Personal Information
    story.append(Paragraph("Personal Information", styles['Heading2']))
    personal_info = data.get('personal_info', {})
    info_text = f"""
    Age: {personal_info.get('age', 'N/A')}<br/>
    Gender: {personal_info.get('gender', 'N/A')}<br/>
    Height: {personal_info.get('height', 'N/A')} cm<br/>
    Weight: {personal_info.get('weight', 'N/A')} kg<br/>
    BMI: {personal_info.get('bmi', 'N/A')}<br/>
    """
    story.append(Paragraph(info_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Nutritional Goals
    story.append(Paragraph("Nutritional Goals", styles['Heading2']))
    goals = data.get('nutritional_goals', {})
    goals_text = f"""
    Daily Calories: {goals.get('calories', 'N/A')}<br/>
    Protein: {goals.get('protein', 'N/A')}g<br/>
    Carbohydrates: {goals.get('carbs', 'N/A')}g<br/>
    Fats: {goals.get('fats', 'N/A')}g<br/>
    Fiber: {goals.get('fiber', 'N/A')}g<br/>
    """
    story.append(Paragraph(goals_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Meal Plan Summary
    story.append(Paragraph("Meal Plan Summary", styles['Heading2']))
    meal_plan = data.get('meal_plan', [])
    if meal_plan:
        meal_data = [['Meal', 'Calories', 'Protein (g)', 'Carbs (g)', 'Fats (g)']]
        for meal in meal_plan[:5]:  # Show first 5 meals
            meal_data.append([
                meal.get('name', 'Unknown'),
                meal.get('calories', 0),
                meal.get('protein', 0),
                meal.get('carbs', 0),
                meal.get('fats', 0)
            ])
        
        meal_table = Table(meal_data)
        meal_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(meal_table)
    
    story.append(Spacer(1, 12))
    
    # Recommendations
    story.append(Paragraph("Recommendations", styles['Heading2']))
    recommendations = data.get('recommendations', [])
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", styles['Normal']))
    
    # Footer
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

if __name__ == '__main__':
    # Create necessary directories and files
    if not os.path.exists('users.json'):
        with open('users.json', 'w') as f:
            json.dump({}, f)
    
    app.run(debug=True, host='0.0.0.0', port=5000)