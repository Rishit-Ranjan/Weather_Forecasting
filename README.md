## Install Dependencies: 
First, install the required libraries using the requirements.txt file created. Open your terminal or command prompt in the project directory and run:
pip install -r requirements.txt

## Set Up Your API Key: 
Replace "your_api_key_here" with your actual OpenWeatherMap API key in .env

## Train the Model: 
To train the model, run the train_model.py script. This will fetch the latest weather data, train the model, and save the necessary files (.pkl files).
python train_model.py

## Make a Prediction: 
Once the model is trained, we can make a new prediction by running the predict_rainfall.py script:
python predict_rainfall.py <br/>

<img width="833" height="221" alt="image" src="https://github.com/user-attachments/assets/e611a6e4-e0e7-462b-8c8f-021eee74bbf4" /><br/><br/>

<img width="1639" height="959" alt="image" src="https://github.com/user-attachments/assets/9182c4d9-c513-4ea1-b3e7-08450f56090f" />

