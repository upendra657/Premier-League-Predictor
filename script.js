// script.js

document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('predict-form');
    const resultDiv = document.getElementById('result');
    const homeTeamSelect = document.getElementById('home-team');
    const awayTeamSelect = document.getElementById('away-team');

    // Prevent selecting same team for home and away
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

    form.addEventListener('submit', async function (e) {
        e.preventDefault();

        const homeTeam = homeTeamSelect.value;
        const awayTeam = awayTeamSelect.value;
        const year = document.getElementById('year').value;

        if (!homeTeam || !awayTeam) {
            resultDiv.innerHTML = '<p style="color: #ff6b6b;">Please select both home and away teams.</p>';
            return;
        }

        if (homeTeam === awayTeam) {
            resultDiv.innerHTML = '<p style="color: #ff6b6b;">Home and Away teams must be different.</p>';
            return;
        }

        resultDiv.innerHTML = '<p>Predicting match outcome...</p>';

        try {
            const response = await fetch('https://premier-league-backend.onrender.com/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam, year: year })
            });

            const data = await response.json();
            
            if (data.error) {
                resultDiv.innerHTML = `<p style="color: #ff6b6b;">Error: ${data.error}</p>`;
            } else {
                const confidence = (data.confidence * 100).toFixed(2);
                const resultColor = data.predicted_label === 'H' ? '#4CAF50' : '#ff6b6b';
                
                resultDiv.innerHTML = `
                    <p><strong>Predicted Outcome:</strong> <span style="color: ${resultColor}">${data.predicted_label}</span></p>
                    <p><strong>Confidence:</strong> ${confidence}%</p>
                    ${data.actual_result ? `<p><strong>Actual Result:</strong> ${data.actual_result}</p>` : ''}
                `;
            }
        } catch (error) {
            resultDiv.innerHTML = `
                <p style="color: #ff6b6b;">Error: Could not connect to the prediction service.</p>
                <p style="font-size: 0.9em; color: #ccc;">Please try again later.</p>
            `;
        }
    });
});