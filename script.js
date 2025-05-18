// script.js

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predict-form');
    const resultContainer = document.getElementById('result');
    const homeTeamSelect = document.getElementById('home-team');
    const awayTeamSelect = document.getElementById('away-team');

    // Prevent selecting the same team for home and away
    homeTeamSelect.addEventListener('change', function() {
        if (this.value === awayTeamSelect.value) {
            awayTeamSelect.value = '';
        }
    });

    awayTeamSelect.addEventListener('change', function() {
        if (this.value === homeTeamSelect.value) {
            homeTeamSelect.value = '';
        }
    });

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading state
        resultContainer.innerHTML = '<div class="loading show">Predicting match outcome...</div>';
        resultContainer.classList.add('show');

        const formData = {
            home_team: homeTeamSelect.value,
            away_team: awayTeamSelect.value,
            year: document.getElementById('year').value
        };

        try {
            const response = await fetch('https://premier-league-predictor-production.up.railway.app/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            if (response.ok) {
                let resultHTML = '';
                
                if (data.predicted_label === 'H') {
                    resultHTML = `
                        <div class="prediction-result">
                            <p>Prediction: ${formData.home_team} will win at home</p>
                            <p class="confidence">Confidence: ${(data.confidence * 100).toFixed(1)}%</p>
                        </div>
                    `;
                } else {
                    resultHTML = `
                        <div class="prediction-result">
                            <p>Prediction: ${formData.away_team} will win away</p>
                            <p class="confidence">Confidence: ${(data.confidence * 100).toFixed(1)}%</p>
                        </div>
                    `;
                }

                if (data.actual_result) {
                    resultHTML += `
                        <div class="actual-result">
                            <p>Actual Result: ${data.actual_result === 'H' ? formData.home_team + ' won' : formData.away_team + ' won'}</p>
                        </div>
                    `;
                }

                resultContainer.innerHTML = resultHTML;
            } else {
                resultContainer.innerHTML = `
                    <div class="error-message">
                        Error: ${data.error || 'Failed to get prediction'}
                    </div>
                `;
            }
        } catch (error) {
            resultContainer.innerHTML = `
                <div class="error-message">
                    Error: Could not connect to the prediction service
                </div>
            `;
        }
    });
});