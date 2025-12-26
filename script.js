document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = '<span class="loading">Calculating...</span>';

    const data = {
        datetime: document.getElementById('datetime').value,
        temperature: document.getElementById('temperature').value,
        humidity: document.getElementById('humidity').value,
        wind_speed: document.getElementById('wind_speed').value,
        pressure: document.getElementById('pressure').value,
        weather_desc: document.getElementById('weather_desc').value
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        
        if (result.error) {
            resultDiv.innerHTML = `<span style="color: red;">Error: ${result.error}</span>`;
        } else {
            resultDiv.innerHTML = `Predicted Rainfall: <span style="color: #007bff;">${result.prediction}</span>`;
        }
    } catch (error) {
        console.error(error);
        resultDiv.innerHTML = `<span style="color: red;">Connection Error</span>`;
    }
});