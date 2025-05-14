from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import requests

app = Flask(__name__)

# Load model and encoders
with open("f1_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Make sure we use the exact values from our encoders for the dropdowns
# Additionally define the 2025 season data
drivers_2025 = [
    "Lando Norris", "Oscar Piastri", "Max Verstappen", "Yuki Tsunoda",
    "Charles Leclerc", "Lewis Hamilton", "George Russell", "Kimi Antonelli",
    "Fernando Alonso", "Lance Stroll", "Pierre Gasly", "Jack Doohan",
    "Carlos Sainz", "Alexander Albon", "Esteban Ocon", "Oliver Bearman",
    "Liam Lawson", "Isack Hadjar", "Nico Hulkenberg", "Gabriel Bortoleto"
]

# Mapping for 2025 constructors to the names in the dataset
constructors_2025_mapping = {
    "McLaren": "McLaren", 
    "Red Bull Racing": "Red Bull",
    "Ferrari": "Ferrari",
    "Mercedes": "Mercedes",
    "Aston Martin": "Aston Martin",
    "Alpine": "Alpine F1 Team",
    "Williams": "Williams",
    "Haas": "Haas F1 Team",
    "Racing Bulls": "AlphaTauri",  # formerly AlphaTauri/RB F1 Team
    "Kick Sauber": "Sauber"  # formerly Alfa Romeo/Sauber
}

constructors_2025 = list(constructors_2025_mapping.keys())

circuits_2025 = [
    "Melbourne", "Shanghai", "Suzuka", "Sakhir", "Jeddah", "Miami",
    "Imola", "Monaco", "Barcelona", "Montreal", "Spielberg", "Silverstone",
    "Spa", "Budapest", "Zandvoort", "Monza", "Baku", "Singapore",
    "Austin", "Mexico City", "Sao Paulo", "Las Vegas", "Lusail", "Yas Marina"
]

# For the original dropdown options
drivers = sorted(list(encoders['driver'].classes_))
constructors = sorted(list(encoders['constructor'].classes_))
circuits = sorted(list(encoders['circuit'].classes_))

# Filter drivers, constructors, and circuits to only include those in 2025 lists that are also in the encoder classes
valid_drivers_2025 = [driver for driver in drivers_2025 if driver in encoders['driver'].classes_]
valid_constructors_2025 = [constructor for constructor in constructors_2025 if constructor in encoders['constructor'].classes_]
valid_circuits_2025 = [circuit for circuit in circuits_2025 if circuit in encoders['circuit'].classes_]

# Create driver-constructor mapping for 2025 season - using surnames
driver_constructor_mapping = {
    "Norris": "McLaren",
    "Piastri": "McLaren",
    "Verstappen": "Red Bull Racing",
    "Tsunoda": "Racing Bulls",
    "Leclerc": "Ferrari",
    "Hamilton": "Ferrari",
    "Russell": "Mercedes",
    "Antonelli": "Mercedes",
    "Alonso": "Aston Martin",
    "Stroll": "Aston Martin",
    "Gasly": "Alpine",
    "Doohan": "Alpine",
    "Sainz": "Williams",
    "Albon": "Williams", 
    "Ocon": "Haas",
    "Bearman": "Haas",
    "Lawson": "Racing Bulls",
    "Hadjar": "Kick Sauber",
    "Hulkenberg": "Kick Sauber",
    "Bortoleto": "Kick Sauber"
}

# Cache file path for circuit predictions
CACHE_FILE = "circuit_predictions_cache.json"
CACHE_EXPIRY = 86400  # Cache expiry in seconds (24 hours)

# Create a directory for storing circuit images
if not os.path.exists('static'):
    os.makedirs('static')
if not os.path.exists('static/circuits'):
    os.makedirs('static/circuits')
    
# Circuit image URLs
CIRCUIT_IMAGES = {
    "Melbourne": "australia",
    "Shanghai": "china",  
    "Suzuka": "japan",
    "Sakhir": "bahrain",
    "Jeddah": "saudi-arabia",
    "Miami": "miami",
    "Imola": "emilia-romagna",
    "Monaco": "monaco",
    "Barcelona": "spain",
    "Montreal": "canada",
    "Spielberg": "austria",
    "Silverstone": "great-britain",
    "Spa": "belgium",
    "Budapest": "hungary",
    "Zandvoort": "netherlands",
    "Monza": "italy",
    "Baku": "azerbaijan",
    "Singapore": "singapore",
    "Austin": "usa",
    "Mexico City": "mexico",
    "Sao Paulo": "brazil",
    "Las Vegas": "las-vegas",
    "Lusail": "qatar",
    "Yas Marina": "abu-dhabi"
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    win_probability = None

    if request.method == "POST":
        race_year = int(request.form["race_year"])
        qualifying_position = int(request.form["qualifying_position"])
        circuit = request.form["circuit"]
        driver = request.form["driver"]
        constructor = request.form["constructor"]

        # Add debugging information
        print(f"Received values - Circuit: {circuit}, Driver: {driver}, Constructor: {constructor}")
        print(f"Available circuits: {list(encoders['circuit'].classes_)[:5]}...")
        print(f"Available drivers: {list(encoders['driver'].classes_)[:5]}...")
        print(f"Available constructors: {list(encoders['constructor'].classes_)[:5]}...")
        
        # Encode input with better error handling
        try:
            # Check if values are in the encoders' vocabulary
            if circuit not in encoders['circuit'].classes_:
                prediction = f"Error: Circuit '{circuit}' is not in the model's training data. Available circuits include: {', '.join(encoders['circuit'].classes_[:5])}..."
                return render_template("index.html", prediction=prediction, drivers=drivers, constructors=constructors, circuits=circuits)
                
            if driver not in encoders['driver'].classes_:
                prediction = f"Error: Driver '{driver}' is not in the model's training data. Available drivers include: {', '.join(encoders['driver'].classes_[:5])}..."
                return render_template("index.html", prediction=prediction, drivers=drivers, constructors=constructors, circuits=circuits)
                
            if constructor not in encoders['constructor'].classes_:
                prediction = f"Error: Constructor '{constructor}' is not in the model's training data. Available constructors include: {', '.join(encoders['constructor'].classes_[:5])}..."
                return render_template("index.html", prediction=prediction, drivers=drivers, constructors=constructors, circuits=circuits)
            
            # If all checks pass, encode the values
            circuit_encoded = encoders["circuit"].transform([circuit])[0]
            driver_encoded = encoders["driver"].transform([driver])[0]
            constructor_encoded = encoders["constructor"].transform([constructor])[0]
            
        except ValueError as e:
            # Log the specific error
            print(f"Encoding error: {str(e)}")
            prediction = f"Encoding error: {str(e)}. Please make sure your selections match the training data."
            return render_template("index.html", prediction=prediction, drivers=drivers, constructors=constructors, circuits=circuits)

        # Prepare input for model
        input_df = pd.DataFrame([{
            "race_year": race_year,
            "qualifying_position": qualifying_position,
            "circuit": circuit_encoded,
            "driver": driver_encoded,
            "constructor": constructor_encoded
        }])

        # Predict
        pred = model.predict(input_df)[0]
        win_probability = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else 0.5
        prediction = "Yes, likely to win!" if pred == 1 else "No, unlikely to win."
    return render_template("index.html", prediction=prediction, win_probability=win_probability, drivers=drivers, constructors=constructors, circuits=circuits)

@app.route("/circuit_predictions")
def circuit_predictions():
    # Always try to load cached predictions first
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            
            print("Using cached predictions")
            return render_template("circuit_predictions.html", circuits=cache_data['predictions'])
        except Exception as e:
            print(f"Error loading cache: {str(e)}")
    
    # If cache doesn't exist or there was an error, generate new predictions
    race_year = 2025
    all_predictions = []
    
    # Debug logging
    print("Available driver encoders:", list(encoders['driver'].classes_)[:5])
    print("Available constructor encoders:", list(encoders['constructor'].classes_)[:5])
    
    # Create a mapping from pretty names to encoder names for circuits if needed
    circuit_name_mapping = {
        "Melbourne": "Albert Park Grand Prix Circuit",
        "Shanghai": "Shanghai International Circuit",
        "Suzuka": "Suzuka Circuit",
        "Sakhir": "Bahrain International Circuit",
        "Jeddah": "Jeddah Corniche Circuit",
        "Miami": "Miami International Autodrome",
        "Imola": "Autodromo Enzo e Dino Ferrari",
        "Monaco": "Circuit de Monaco",
        "Barcelona": "Circuit de Barcelona-Catalunya",
        "Montreal": "Circuit Gilles Villeneuve",
        "Spielberg": "Red Bull Ring",
        "Silverstone": "Silverstone Circuit",
        "Spa": "Circuit de Spa-Francorchamps",
        "Budapest": "Hungaroring",
        "Zandvoort": "Circuit Zandvoort",
        "Monza": "Autodromo Nazionale di Monza",
        "Baku": "Baku City Circuit",
        "Singapore": "Marina Bay Street Circuit",
        "Austin": "Circuit of the Americas",
        "Mexico City": "Autódromo Hermanos Rodríguez",
        "Sao Paulo": "Autódromo José Carlos Pace",
        "Las Vegas": "Las Vegas Strip Circuit",
        "Lusail": "Lusail International Circuit",
        "Yas Marina": "Yas Marina Circuit"
    }
    
    # Get a list of circuits that are actually in our training data
    for display_name, encoder_name in circuit_name_mapping.items():
        circuit_results = []
        circuit_name_to_use = None
        
        # Check which name exists in our encoder
        if encoder_name in encoders['circuit'].classes_:
            circuit_name_to_use = encoder_name
            print(f"Using encoder name for {display_name}: {encoder_name}")
        elif display_name in encoders['circuit'].classes_:
            circuit_name_to_use = display_name
            print(f"Using display name for {display_name}")
        else:
            # Skip if neither name is in our encoder
            print(f"Skipping {display_name} - not found in encoder")
            continue
        
        # Encode the circuit
        circuit_encoded = encoders["circuit"].transform([circuit_name_to_use])[0]
        
        # Check each driver with this circuit
        for driver_full_name in drivers_2025:
            # Try different ways to match the driver in our encoder
            driver_to_use = None
            
            # Try with full name
            if driver_full_name in encoders['driver'].classes_:
                driver_to_use = driver_full_name
                print(f"Found driver full name: {driver_full_name}")
            else:
                # Try with just the surname
                surname = driver_full_name.split()[-1]
                if surname in encoders['driver'].classes_:
                    driver_to_use = surname
                    print(f"Found driver surname: {surname} from {driver_full_name}")
                else:
                    print(f"Skipping driver {driver_full_name} - not found in encoder")
                    continue
            
            # Get constructor for this driver and map to dataset name
            surname = driver_full_name.split()[-1]
            team_2025_name = driver_constructor_mapping.get(surname)
            
            if not team_2025_name:
                print(f"Skipping driver {driver_full_name} - no 2025 constructor mapping")
                continue
                
            # Map the 2025 team name to the dataset team name
            constructor = constructors_2025_mapping.get(team_2025_name)
            if not constructor or constructor not in encoders['constructor'].classes_:
                # Try the team name directly in case it's already in encoder format
                if team_2025_name in encoders['constructor'].classes_:
                    constructor = team_2025_name
                else:
                    print(f"Skipping driver {driver_full_name} - constructor {team_2025_name} not found in encoder")
                    continue
            
            print(f"Processing {driver_to_use} with {constructor} at {display_name}")
            
            # Get best qualifying position for this driver at this circuit
            best_qualifying = None
            best_probability = 0
            
            # For each qualifying position 1-10
            for qualifying_position in range(1, 11):
                # Encode driver and constructor
                try:
                    driver_encoded = encoders["driver"].transform([driver_to_use])[0]
                    constructor_encoded = encoders["constructor"].transform([constructor])[0]
                except Exception as e:
                    print(f"Encoding error for {driver_to_use}/{constructor}: {str(e)}")
                    continue
                
                # Prepare input for model
                input_df = pd.DataFrame([{
                    "race_year": race_year,
                    "qualifying_position": qualifying_position,
                    "circuit": circuit_encoded,
                    "driver": driver_encoded,
                    "constructor": constructor_encoded
                }])
                
                # Predict win probability
                try:
                    prediction = model.predict(input_df)[0]
                    win_probability = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else 0.5
                    
                    # If this is a win and better than previous qualifying position
                    if prediction == 1 and win_probability > best_probability:
                        best_qualifying = qualifying_position
                        best_probability = win_probability
                        print(f"Found win for {driver_to_use} at {display_name}: qual={qualifying_position}, prob={win_probability:.2f}")
                        
                except Exception as e:
                    print(f"Error predicting for {driver_to_use} at {display_name}: {str(e)}")
            
            # If we found a winning qualifying position for this driver
            if best_qualifying is not None:
                circuit_results.append({
                    'driver': driver_full_name,
                    'constructor': team_2025_name,  # Use the 2025 team name for display
                    'qualifying': best_qualifying,
                    'win_probability': best_probability,
                    'prediction': "Win"
                })
        
        # Sort by win probability
        circuit_results.sort(key=lambda x: x['win_probability'], reverse=True)
        print(f"Circuit {display_name} has {len(circuit_results)} results")
        
        # Add circuit to predictions if we have results
        if circuit_results:
            all_predictions.append({
                'circuit': display_name,
                'results': circuit_results[:5]  # Top 5 likely winners
            })
    
    print(f"Total predictions: {len(all_predictions)} circuits")
    
    # Cache the predictions
    try:
        cache_data = {
            'timestamp': datetime.now().timestamp(),
            'predictions': all_predictions
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)
        print("Predictions cached successfully")
    except Exception as e:
        print(f"Error caching predictions: {str(e)}")
    
    return render_template("circuit_predictions.html", circuits=all_predictions)

@app.route('/circuit_image/<circuit_name>')
def circuit_image(circuit_name):
    # Map circuit name to its image code
    image_code = CIRCUIT_IMAGES.get(circuit_name)
    if not image_code:
        # Use the circuit name as a fallback
        image_code = circuit_name.lower().replace(' ', '-')
    
    # Path to save the image
    image_path = f'static/circuits/{image_code}.png'
    
    # If image doesn't exist locally, try to download it
    if not os.path.exists(image_path):
        try:
            url = f"https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/{image_code}_Circuit.png.transform/9col/image.png"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    f.write(response.content)
            else:
                # Use a default image if the circuit image is not available
                image_path = 'static/circuits/default.png'
                if not os.path.exists(image_path):
                    # Create a very simple default image if it doesn't exist
                    with open(image_path, 'wb') as f:
                        f.write(b'')  # Empty file as a placeholder
        except Exception as e:
            print(f"Error downloading circuit image: {str(e)}")
            image_path = 'static/circuits/default.png'
    
    # Return the image URL
    return jsonify({'url': f'/{image_path}'})

if __name__ == "__main__":
    app.run(debug=True)
