function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    
    document.getElementById(tabName + '-tab').style.display = 'block';
    event.target.classList.add('active');
}

async function handlePredict(event, mode) {
    event.preventDefault();
    const resultDiv = document.getElementById('result');
    const resultText = document.getElementById('prediction-text');
    resultDiv.classList.remove('hidden');
    resultText.textContent = "Predicting...";
    resultText.className = "";

    let payload = {};

    if (mode === 'location') {
        payload = {
            lat: document.getElementById('lat').value,
            lon: document.getElementById('lon').value
        };
    } else {
        payload = {
            datetime: document.getElementById('datetime').value,
            temperature: document.getElementById('temperature').value,
            humidity: document.getElementById('humidity').value,
            wind_speed: document.getElementById('wind_speed').value,
            pressure: document.getElementById('pressure').value,
            weather_desc: document.getElementById('weather_desc').value
        };
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (response.ok) {
            resultText.innerHTML = `
                <strong>Rainfall (3h):</strong> ${data.prediction_mm} mm<br>
                <strong>Rainfall (in):</strong> ${data.prediction_inches} in
            `;
        } else {
            resultText.textContent = "Error: " + (data.error || "Unknown error");
            resultText.className = "error";
        }
    } catch (error) {
        resultText.textContent = "Connection Error: " + error.message;
        resultText.className = "error";
    }
}

document.getElementById('location-form').addEventListener('submit', (e) => handlePredict(e, 'location'));
document.getElementById('manual-form').addEventListener('submit', (e) => handlePredict(e, 'manual'));