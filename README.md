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

<img width="989" height="340" alt="Screenshot 2025-09-11 181857" src="https://github.com/user-attachments/assets/15b05eee-109e-44c2-af0f-3c26e496ad41" />
