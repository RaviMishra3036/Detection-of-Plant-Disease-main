import pickle
try:
    import cv2
    import numpy as np
    from tensorflow.keras.preprocessing.image import img_to_array
except ImportError:
    cv2 = None
    np = None
    img_to_array = None
import json
import base64
import mimetypes
import os
import re
from pathlib import Path
from uuid import uuid4
from datetime import date, datetime, timedelta
from urllib.parse import quote_plus, urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Constants & Paths
DEFAULT_IMAGE_SIZE = (256, 256)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'plant_disease_classification_model.pkl'
LABEL_PATH = BASE_DIR / 'plant_disease_label_transform.pkl'
UPLOAD_DIR = Path(os.getenv('UPLOAD_DIR', '/tmp' if os.getenv('VERCEL') else str(BASE_DIR / 'uploads')))
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
MIN_CONFIDENCE = 0.45
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-3.6-flash')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

CROP_PROFILES = {
    'apple': {'name': 'Apple', 'harvest_days': 160, 'soil': 'Well-drained loam; pH 6.0–7.0', 'water': 'Keep root zone evenly moist; avoid wet leaves.', 'trained': True},
    'tomato': {'name': 'Tomato', 'harvest_days': 85, 'soil': 'Fertile, well-drained loam; pH 6.0–6.8', 'water': 'Water at the soil in the morning; avoid leaf wetness.', 'trained': True},
    'cherry': {'name': 'Cherry', 'harvest_days': 130, 'soil': 'Deep, well-drained loam; pH 6.0–7.0', 'water': 'Water deeply during dry periods; ensure drainage.', 'trained': True},
    'grape': {'name': 'Grape', 'harvest_days': 150, 'soil': 'Well-drained soil; pH 5.5–7.0', 'water': 'Avoid standing water; improve airflow through canopy.', 'trained': True},
    'peach': {'name': 'Peach', 'harvest_days': 130, 'soil': 'Sandy loam with good drainage; pH 6.0–7.0', 'water': 'Deep, infrequent watering is better than shallow daily watering.', 'trained': True},
    'strawberry': {'name': 'Strawberry', 'harvest_days': 90, 'soil': 'Organic, well-drained soil; pH 5.5–6.5', 'water': 'Keep soil moist and mulch fruit away from wet soil.', 'trained': True},
    'potato': {'name': 'Potato', 'harvest_days': 110, 'soil': 'Loose, well-drained soil; pH 5.0–6.5', 'water': 'Moist, never waterlogged; reduce water before harvest.', 'trained': False},
    'mango': {'name': 'Mango', 'harvest_days': 150, 'soil': 'Deep, well-drained soil; pH 5.5–7.5', 'water': 'Water young trees regularly; do not let water stand.', 'trained': False},
    'herb': {'name': 'Herb / other crop', 'harvest_days': 60, 'soil': 'Use well-drained soil and confirm crop-specific pH needs.', 'water': 'Check soil before watering; do not water on a fixed schedule.', 'trained': False},
}

WEATHER_CODES = {
    0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
    45: 'Fog', 48: 'Rime fog', 51: 'Light drizzle', 53: 'Drizzle', 55: 'Heavy drizzle',
    61: 'Light rain', 63: 'Rain', 65: 'Heavy rain', 71: 'Light snow', 80: 'Rain showers',
    81: 'Moderate showers', 82: 'Heavy showers', 95: 'Thunderstorm'
}

# Load the local model when it is available. Vercel uses the Gemini path without
# bundling this large optional asset.
model = None
image_labels = None
if MODEL_PATH.exists() and LABEL_PATH.exists() and np is not None:
    print("[INFO] Loading Model & Labels for Web App...")
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(LABEL_PATH, 'rb') as label_file:
        image_labels = pickle.load(label_file)


def load_local_env():
    """Load simple KEY=value entries without adding a package dependency."""
    env_path = BASE_DIR / '.env'
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding='utf-8').splitlines():
        if '=' not in line or line.lstrip().startswith('#'):
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()

def convert_image_to_array(image_dir):
    if cv2 is None or img_to_array is None:
        return None
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, DEFAULT_IMAGE_SIZE)
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error: {e}")
        return None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def format_label(label):
    return label.replace('___', ' — ').replace('_', ' ')


def request_json(endpoint, params):
    query = urlencode(params)
    with urlopen(f'{endpoint}?{query}', timeout=6) as response:
        return json.loads(response.read().decode('utf-8'))


def get_weather(location):
    if not location:
        return None, None
    try:
        geocode = request_json('https://geocoding-api.open-meteo.com/v1/search', {
            'name': location, 'count': 1, 'language': 'en', 'format': 'json'
        })
        places = geocode.get('results', [])
        if not places:
            return None, 'Location not found. Try city and state/district.'
        place = places[0]
        forecast = request_json('https://api.open-meteo.com/v1/forecast', {
            'latitude': place['latitude'], 'longitude': place['longitude'], 'timezone': 'auto',
            'current': 'temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,soil_temperature_0cm,soil_moisture_0_to_1cm'
        })
        current = forecast.get('current', {})
        return {
            'place': f"{place['name']}, {place.get('admin1') or place.get('country', '')}".strip(', '),
            'temperature': current.get('temperature_2m'),
            'humidity': current.get('relative_humidity_2m'),
            'rain': current.get('precipitation'),
            'wind': current.get('wind_speed_10m'),
            'soil_temperature': current.get('soil_temperature_0cm'),
            'soil_moisture': current.get('soil_moisture_0_to_1cm'),
            'condition': WEATHER_CODES.get(current.get('weather_code'), 'Current conditions'),
            'updated_at': current.get('time'),
        }, None
    except Exception:
        return None, 'Live weather is temporarily unavailable. Your crop plan is still available.'


def build_crop_plan(crop_key, planted_on, soil_ph, manual_soil_temp):
    crop = CROP_PROFILES.get(crop_key, CROP_PROFILES['herb'])
    harvest_date = None
    if planted_on:
        try:
            harvest_date = (datetime.strptime(planted_on, '%Y-%m-%d').date() + timedelta(days=crop['harvest_days'])).strftime('%d %b %Y')
        except ValueError:
            pass
    notes = [crop['water']]
    if soil_ph:
        try:
            value = float(soil_ph)
            if value < 5.5 or value > 7.5:
                notes.append('Your entered pH is outside the common range for most listed crops; verify with a soil test before amending.')
        except ValueError:
            pass
    if manual_soil_temp:
        try:
            value = float(manual_soil_temp)
            if value < 12:
                notes.append('Soil is cool. Avoid planting warm-season crops until crop-specific soil-temperature requirements are met.')
        except ValueError:
            pass
    return {'crop': crop, 'harvest_date': harvest_date, 'notes': notes}


def build_price_links(medicine_options, crop_name):
    """Create transparent search links; no price or seller claim is made by the app."""
    links = []
    for option in medicine_options[:3]:
        ingredient = str(option.get('active_ingredient', '')).strip()
        if not ingredient:
            continue
        query = quote_plus(f'{ingredient} {crop_name} plant disease treatment')
        links.append({
            'name': ingredient,
            'search': f'https://www.google.com/search?q={query}+price+India',
            'amazon': f'https://www.amazon.in/s?k={query}',
            'buyhatke': 'https://price.buyhatke.com/',
        })
    return links


def build_analysis_prompt(crop_plan, weather):
    crop_name = crop_plan['crop']['name']
    weather_context = 'No live weather was requested.'
    if weather:
        weather_context = (
            f"{weather['place']}: {weather['condition']}, air {weather['temperature']} C, "
            f"humidity {weather['humidity']}%, soil estimate {weather['soil_temperature']} C, "
            f"soil moisture estimate {weather['soil_moisture']} m3/m3."
        )
    return f'''You are a cautious agricultural image-screening assistant. Analyse this image for a farmer.
Selected crop: {crop_name}. Weather context: {weather_context}
Return ONLY valid JSON, with exactly these keys:
plant_or_object, crop_match, health_status, confidence, observations, possible_conditions, immediate_actions, care_guidance, water_guidance, nutrition_guidance, temperature_guidance, harvest_guidance, medicine_options, hindi_summary, english_summary, safety_note.
Rules:
- Do not invent certainty. If the photo is not a plant/leaf, say so in plant_or_object and set health_status to "not assessable".
- health_status must be one of "healthy-looking", "possible issue", "uncertain", "not assessable".
- confidence is an integer from 0 to 100 for visual assessment only, not a lab diagnosis.
- possible_conditions is an array of up to 3 objects with name, likelihood (low/medium/high), and reason.
- immediate_actions and care_guidance are arrays of short practical steps. Do not recommend pesticide brand names or exact chemical doses.
- water_guidance, nutrition_guidance, and temperature_guidance are short, practical strings based on image and supplied weather context; say when data is insufficient.
- medicine_options is an array of up to 3 objects with active_ingredient, purpose, and caution. Include only a generic biological option or a crop/disease-appropriate active ingredient; do not give dose, brand, or a purchase instruction. If diagnosis is uncertain, return an empty array.
- hindi_summary is a simple Hindi farmer-facing summary in Devanagari. english_summary is the same advice in simple English.
- harvest_guidance must say whether the image can or cannot determine harvest readiness.
- safety_note must advise local agronomy confirmation for possible disease or treatment decisions.'''


def parse_analysis(text):
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    analysis = json.loads(text)
    required = {'plant_or_object', 'crop_match', 'health_status', 'confidence', 'observations',
                'possible_conditions', 'immediate_actions', 'care_guidance', 'water_guidance',
                'nutrition_guidance', 'temperature_guidance', 'harvest_guidance', 'medicine_options',
                'hindi_summary', 'english_summary', 'safety_note'}
    if not required.issubset(analysis):
        raise ValueError('Incomplete structured response')
    return analysis


def openai_compatible_analysis(provider, image_data, mime_type, prompt):
    """Call a vision model using the OpenAI-compatible API used by several providers."""
    settings = {
        'grok': ('XAI_API_KEY', 'XAI_BASE_URL', 'XAI_MODEL', 'https://api.x.ai/v1/chat/completions', 'grok-2-vision-1212'),
        'openrouter': ('OPENROUTER_API_KEY', 'OPENROUTER_BASE_URL', 'OPENROUTER_MODEL', 'https://openrouter.ai/api/v1/chat/completions', 'google/gemini-2.5-flash'),
        'huggingface': ('HF_TOKEN', 'HF_BASE_URL', 'HF_MODEL', 'https://router.huggingface.co/v1/chat/completions', 'google/gemma-3-4b-it'),
        'nvidia': ('NVIDIA_API_KEY', 'NVIDIA_BASE_URL', 'NVIDIA_MODEL', 'https://integrate.api.nvidia.com/v1/chat/completions', 'meta/llama-3.2-11b-vision-instruct'),
    }
    key_name, url_name, model_name, default_url, default_model = settings[provider]
    api_key = ''.join(os.getenv(key_name, '').split())
    if not api_key:
        raise ValueError(f'{key_name} is not configured')
    payload = {'model': os.getenv(model_name, default_model), 'temperature': 0.2,
               'messages': [{'role': 'user', 'content': [
                   {'type': 'text', 'text': prompt},
                   {'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{image_data}'}}
               ]}]}
    if provider == 'openrouter':
        payload['response_format'] = {'type': 'json_object'}
    request = Request(os.getenv(url_name, default_url), data=json.dumps(payload).encode('utf-8'),
                      headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}, method='POST')
    with urlopen(request, timeout=35) as response:
        data = json.loads(response.read().decode('utf-8'))
    return parse_analysis(data['choices'][0]['message']['content'])


def ollama_plant_analysis(image_data, mime_type, prompt):
    """Use a local Ollama vision model when every remote provider is unavailable."""
    payload = {'model': os.getenv('OLLAMA_MODEL', 'llama3.2-vision'), 'stream': False,
               'format': 'json', 'images': [image_data], 'prompt': prompt}
    request = Request(os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434/api/generate'),
                      data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=300) as response:
        data = json.loads(response.read().decode('utf-8'))
    return parse_analysis(data['response'])


def gemini_plant_analysis(image_path, crop_plan, weather):
    """Try configured vision providers in order, ending with local Ollama."""
    mime_type = mimetypes.guess_type(str(image_path))[0] or 'image/jpeg'
    image_data = base64.b64encode(Path(image_path).read_bytes()).decode('ascii')
    prompt = build_analysis_prompt(crop_plan, weather)
    failures = []
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        payload = {'contents': [{'parts': [
            {'inline_data': {'mime_type': mime_type, 'data': image_data}},
            {'text': prompt}
        ]}], 'generationConfig': {'temperature': 0.2, 'responseMimeType': 'application/json'}}
        request = Request(f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent',
                          data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json', 'x-goog-api-key': api_key}, method='POST')
        try:
            with urlopen(request, timeout=35) as response:
                data = json.loads(response.read().decode('utf-8'))
            return parse_analysis(data['candidates'][0]['content']['parts'][0]['text']), 'Gemini'
        except Exception as error:
            failures.append(f'Gemini: {str(error)[:100]}')
    else:
        failures.append('Gemini: GEMINI_API_KEY is not configured')

    for provider in ('grok', 'openrouter', 'huggingface', 'nvidia'):
        try:
            return openai_compatible_analysis(provider, image_data, mime_type, prompt), provider.title()
        except Exception as error:
            failures.append(f'{provider.title()}: {str(error)[:100]}')
    ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434/api/generate')
    if os.getenv('VERCEL') or ollama_url.startswith(('http://127.0.0.1', 'http://localhost')):
        failures.append('Ollama is not reachable from this deployment')
    else:
        try:
            return ollama_plant_analysis(image_data, mime_type, prompt), 'Ollama (offline)'
        except Exception:
            failures.append('Ollama request failed')
    return None, 'All configured AI providers are unavailable. Check provider quota, model access, and Ollama deployment settings.'

@app.route('/', methods=['GET'])
def home():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def predict():
    crop_key = request.form.get('crop', 'herb')
    crop_plan = build_crop_plan(crop_key, request.form.get('planted_on', ''), request.form.get('soil_ph', ''), request.form.get('soil_temp', ''))
    weather, weather_error = get_weather(request.form.get('location', '').strip())
    template_args = {'crop_plan': crop_plan, 'weather': weather, 'weather_error': weather_error, 'selected_crop': crop_key}
    if 'imagefile' not in request.files:
        return render_template('upload.html', error='Please choose a leaf image first.', **template_args)
    
    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        return render_template('upload.html', error='Please choose a leaf image first.', **template_args)
    if not allowed_file(imagefile.filename):
        return render_template('upload.html', error='Please upload a JPG, PNG, or WEBP image.', **template_args)

    UPLOAD_DIR.mkdir(exist_ok=True)
    filename = secure_filename(imagefile.filename)
    image_path = UPLOAD_DIR / f'{uuid4().hex}_{filename}'
    imagefile.save(image_path)

    image_array = convert_image_to_array(str(image_path))
    crop = crop_plan['crop']
    gemini_analysis, gemini_error = gemini_plant_analysis(image_path, crop_plan, weather)
    image_path.unlink(missing_ok=True)
    if gemini_analysis:
        gemini_analysis['provider'] = gemini_error
        gemini_analysis['shopping_links'] = build_price_links(
            gemini_analysis.get('medicine_options', []), crop['name']
        )
        return render_template('upload.html', gemini_analysis=gemini_analysis, **template_args)

    if model is None or image_labels is None or image_array is None or image_array.size == 0:
        return render_template('upload.html', diagnosis_notice=(
            f"Online analysis and the local {crop['name']} model are currently unavailable. "
            "Please try again later or verify the deployment's AI environment settings."
        ), **template_args)

    # Process and predict with the optional local model.
    np_image = np.array(image_array, dtype=np.float16) / 255.0
    np_image = np.expand_dims(np_image, axis=0)

    # Offline fallback: this model only understands a small, local set of crop diseases.
    preds = model.predict(np_image, verbose=0)
    confidence_scores = preds[0]
    trained_labels = [(i, label) for i, label in enumerate(image_labels.classes_)
                      if label.lower().startswith(crop_key + '___')]

    if not trained_labels:
        return render_template('upload.html', diagnosis_notice=(
            f"Image disease detection is not trained for {crop['name']} yet. "
            "Please choose a supported crop or try again later."
        ), **template_args)

    result, predicted_label = max(trained_labels, key=lambda item: confidence_scores[item[0]])
    confidence = round(float(confidence_scores[result]) * 100, 1)
    if confidence_scores[result] < MIN_CONFIDENCE:
        return render_template('upload.html', diagnosis_notice=(
            f"No reliable {crop['name']} disease match was found ({confidence}% confidence). "
            "Please use a clearer image or confirm the result with a local agriculture expert."
        ), confidence=confidence, **template_args)

    return render_template('upload.html', prediction=format_label(predicted_label), confidence=confidence, **template_args)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
