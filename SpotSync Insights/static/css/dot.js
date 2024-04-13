    function getRandomInt(min, max) {
            return Math.floor(Math.random() * (max - min + 1)) + min;
        }

        function createFallingDot() {
            var dot = document.createElement('div');
            dot.className = 'falling-dot';
            dot.style.left = getRandomInt(1, 100) + 'vw';
            dot.style.top = getRandomInt(1, 100) + 'vh';
            dot.style.animation = 'fall ' + getRandomInt(2, 6) + 's linear infinite';

            dot.addEventListener('animationiteration', function() {
                // Dot's animation cycle is completed, replace it
                dot.style.left = getRandomInt(1, 100) + 'vw';
                dot.style.top = getRandomInt(1, 100) + 'vh';
            });

            document.body.appendChild(dot);
        }

        // Create initial falling dots
        for (var i = 0; i < 25; i++) {
            createFallingDot();
        }


 