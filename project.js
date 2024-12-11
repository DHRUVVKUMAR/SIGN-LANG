document.addEventListener('DOMContentLoaded', () => {
    const startRecognitionButton = document.getElementById('start-recognition');
    const detectedSignSpan = document.getElementById('detected-sign');
    const tutorialButton = document.getElementById('tutorial-button');
    const scrollInstruction = document.getElementById('scroll-instruction');
    const getStartedSection = document.getElementById('get-started');

    startRecognitionButton.addEventListener('click', () => {
        // Simulating recognition start
        startRecognitionButton.textContent = 'Stop Recognition';
        detectedSignSpan.textContent = 'Scanning...';
        
        // In a real application, you would start the camera and recognition process here
    });

    tutorialButton.addEventListener('click', () => {
        alert('Tutorial functionality not implemented in this version.');
    });

    // Check scroll position to show/hide instruction
    function checkScroll() {
        if (getStartedSection.getBoundingClientRect().left < window.innerWidth) {
            scrollInstruction.style.display = 'none';
        } else {
            scrollInstruction.style.display = 'block';
        }
    }

    // Listen for scroll events
    document.querySelector('.project-content').addEventListener('scroll', checkScroll);

    // Initial check
    checkScroll();
});