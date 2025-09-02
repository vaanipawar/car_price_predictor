from flask import Flask, render_template,jsonify,request
import pandas as pd
import pickle


app=Flask(__name__)
car=pd.read_csv("Cleaned Car.csv")

model = pickle.load(open("LinearRegressionModel.pkl",'rb'))

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(),reverse=True)
    fuel_type = car['fuel_type'].unique()

    return render_template('index.html',companies=companies,car_models=car_models,years=year, fuel_types=fuel_type )
@app.route('/get_models/<company>')
def get_models(company):
    try:
        
        models = car[car['company'] == company]['name'].unique()
        models = sorted(models.tolist())
        return jsonify(models)
    except Exception as e:
        return jsonify([])
    
@app.route('/predict', methods=['POST'])    
def predict():
    try:
        
        print("All form data:", dict(request.form))
        
        
        company = request.form.get('company')
        car_model = request.form.get('car_models')  # Matches the form field name
        year = request.form.get('year')
        fuel_type = request.form.get('fuel_type')
        kms_driven = request.form.get('kilo_driven')
        
        print("Raw values:", company, car_model, year, fuel_type, kms_driven)
        
        
        if not all([company, car_model, year, fuel_type, kms_driven]):
            return "Error: All fields are required", 400
        
        
        year = int(year)
        kms_driven = int(kms_driven)
        
        print("Processed values:", company, year, car_model, fuel_type, kms_driven)
        
        
        input_data = pd.DataFrame({
            'company': [company],
            'year': [year],
            'name': [car_model],
            'kms_driven': [kms_driven],
            'fuel_type': [fuel_type]
        })
        
        print("Input data:", input_data)
        
        
        prediction = model.predict(input_data)
        print("Prediction:", prediction)
        
        
        return str(round(prediction[0], 2))
        
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {str(e)}", 400

if __name__ == "__main__":
    app.run(debug=True)
