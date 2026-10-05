document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('sentimentForm');
    const tweetInput = document.getElementById('tweetInput');
    const resultDiv = document.getElementById('result');
    const sentimentOutput = document.getElementById('sentimentText');
    const sentimentEmoji = document.getElementById('sentimentEmoji');
    const confidenceValue = document.getElementById('confidenceValue');
    const exampleButtons = document.querySelectorAll('.example-btn');
    const analyticsContent = document.getElementById('analyticsContent');

    // Handle form submission
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        const text = tweetInput.value.trim();

        if (!text) {
            alert('Please enter some text');
            return;
        }

        // Show loading state
        const originalButtonText = form.querySelector('button').textContent;
        form.querySelector('button').textContent = 'Analyzing...';
        form.querySelector('button').disabled = true;

        // Send request to Flask backend
        fetch('/predict', {
            method: 'POST',
            body: new URLSearchParams({ 'text': text })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
            } else {
                // Display results
                sentimentOutput.textContent = data.sentiment;
                sentimentEmoji.textContent = data.emoji;
                confidenceValue.textContent = data.confidence;
                resultDiv.style.display = 'block';
            }
        })
        .catch(error => {
            alert('Error: ' + error.message);
        })
        .finally(() => {
            // Reset button state
            form.querySelector('button').textContent = originalButtonText;
            form.querySelector('button').disabled = false;
        });
    });

    // Handle example button clicks
    exampleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const text = this.getAttribute('data-text');
            tweetInput.value = text.slice(1, -1); // Remove quotes
            tweetInput.focus();
        });
    });

    // Load analytics data
    function loadAnalytics() {
        fetch('/results/evaluation.txt')
        .then(response => {
            if (!response.ok) throw new Error('Unable to load analytics');
            return response.text();
        })
        .then(text => {
            const lines = text.trim().split('\n');
            analyticsContent.innerHTML = '';

            lines.forEach(line => {
                const [metric, value] = line.split(': ');
                if (metric && value) {
                    const card = document.createElement('div');
                    card.className = 'analytics-card';
                    card.innerHTML = `
                        <h3>${metric}</h3>
                        <p>${value}</p>
                    `;
                    analyticsContent.appendChild(card);
                }
            });
        })
        .catch(error => {
            analyticsContent.innerHTML = '<p>Analytics data not available. Please train the model first.</p>';
        });
    }

    // Load analytics on page load
    loadAnalytics();
});