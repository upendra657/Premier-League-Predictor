// API URL - Update this with your Vercel deployment URL
const API_URL = 'https://premier-league-predictor.vercel.app';

document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('predict-form');
    const resultDiv = document.getElementById('result');
    const homeTeamSelect = document.getElementById('home-team');
    const awayTeamSelect = document.getElementById('away-team');

    // Load teams from API
    async function loadTeams() {
        try {
            const response = await fetch(`${API_URL}/`);
            const data = await response.json();
            if (data.teams && data.teams.length > 0) {
                const teams = data.teams;
                teams.forEach(team => {
                    homeTeamSelect.add(new Option(team, team));
                    awayTeamSelect.add(new Option(team, team));
                });
            }
        } catch (error) {
            console.error('Error loading teams:', error);
            resultDiv.innerHTML = '<p style="color:red;">Error loading teams. Please try again later.</p>';
        }
    }

    // Load teams when page loads
    loadTeams();
  
    form.addEventListener('submit', async function (e) {
        e.preventDefault();
  
        const homeTeam = homeTeamSelect.value;
        const awayTeam = awayTeamSelect.value;
        const year = document.getElementById('year').value;
  
        if (homeTeam === awayTeam) {
            resultDiv.innerHTML = '<p style="color:red;">Home and Away teams must be different.</p>';
            return;
        }
  
        resultDiv.innerHTML = 'Predicting...';
  
        try {
            const response = await fetch(`${API_URL}/predict`, {
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