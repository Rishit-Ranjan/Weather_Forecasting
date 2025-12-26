document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('result');
    const predictionOutput = document.getElementById('prediction-output');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const features = [
            parseFloat(document.getElementById('sunshine').value),
            parseFloat(document.getElementById('humidity9am').value),
            parseFloat(document.getElementById('humidity3pm').value),
            parseFloat(document.getElementById('pressure9am').value),
            parseFloat(document.getElementById('pressure3pm').value),
            parseFloat(document.getElementById('cloud9am').value),
            parseFloat(document.getElementById('cloud3pm').value),
            parseFloat(document.getElementById('temp9am').value),
            parseFloat(document.getElementById('temp3pm').value),
            parseFloat(document.getElementById('raintoday').value),
        ];

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ features: features }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();

            if (data.error) {
                predictionOutput.textContent = `Error: ${data.error}`;
            } else {
                predictionOutput.textContent = data.prediction;
            }

            resultDiv.classList.remove('hidden');

        } catch (error) {
            predictionOutput.textContent = `An error occurred: ${error.message}`;
            resultDiv.classList.remove('hidden');
        }
    });
});
