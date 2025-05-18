// script.js

document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('predict-form');
    const resultDiv = document.getElementById('result');
  
    form.addEventListener('submit', async function (e) {
      e.preventDefault();
  
      const homeTeam = document.getElementById('home-team').value;
      const awayTeam = document.getElementById('away-team').value;
      const year = document.getElementById('year').value;
  
      if (homeTeam === awayTeam) {
        resultDiv.innerHTML = '<p style="color:red;">Home and Away teams must be different.</p>';
        return;
      }
  
      resultDiv.innerHTML = 'Predicting...';
  
      try {
        const response = await fetch('https://your-backend-url.onrender.com/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam, year: year })
        });
  
        const data = await response.json();
        if (data.error) {
          resultDiv.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
        } else {
          resultDiv.innerHTML = `
            <p><strong>Predicted Outcome:</strong> ${data.predicted_label}</p>
            <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
            ${data.actual_result ? `<p><strong>Actual Result:</strong> ${data.actual_result}</p>` : ''}
          `;
        }
      } catch (error) {
        resultDiv.innerHTML = `<p style="color:red;">Request failed: ${error.message}</p>`;
      }
    });
  });