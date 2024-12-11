const developers = [
    {
        name: 'Debasmita Dhar',
        image: 'Developers/deba.jpeg?height=200&width=200',
        github: 'https://github.com/debasmitaa2907',
        linkedin: 'https://linkedin.com/in/debasmita-dhar'
    },
    {
        name: 'Parth Patel',
        image: 'Developers/parth.jpg?height=200&width=200',
        github: 'https://github.com/Parthp1205',
        linkedin: 'https://linkedin.com/in/parthp1205/'
    },
    {
        name: 'Dhruv Kumar',
        image: 'Developers/dhruv.jpeg?height=200&width=200',
        github: 'https://github.com/DHRUVVKUMAR',
        linkedin: 'https://www.linkedin.com/in/dhruv-kumar-bb348625b/'
    },
    {
        name: 'Samangya Nayak',
        image: 'Developers/kuki.jpeg?height=200&width=200',
        github: 'https://github.com/samangya',
        linkedin: 'https://linkedin.com/in/samangyanayak'
    },
    {
        name: 'Shashwat Mishra',
        image: 'Developers/necro.jpeg?height=200&width=200',
        github: 'https://github.com/NECRO-0',
        linkedin: 'https://www.linkedin.com/in/shashwat-mishra-a295ba22a/'
    },
    {
        name: 'Arpita Datta ',
        image: 'Developers/arpita.jpeg?height=200&width=200',
        github: 'https://github.com/ArpitaDatta23',
        linkedin: 'https://linkedin.com/in/arpitadatta23'
    }
];

let currentDeveloper = 0;
let isScrolling = false;

// Function to update developer profile
function updateDeveloperProfile() {
    const developer = developers[currentDeveloper];
    const profile = document.getElementById('developer-profile');
    
    profile.style.opacity = '0';
    profile.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        // Update developer profile image and name
        document.getElementById('developer-image').src = developer.image;
        document.getElementById('developer-name').textContent = developer.name;
        
        // Update the GitHub and LinkedIn links
        const githubLink = document.getElementById('github-link');
        const linkedinLink = document.getElementById('linkedin-link');

        // Ensure the href values are properly set
        githubLink.href = developer.github;
        linkedinLink.href = developer.linkedin;

        // Debugging log to ensure hrefs are set
        console.log('GitHub URL:', githubLink.href);
        console.log('LinkedIn URL:', linkedinLink.href);

        // Make the profile visible after the update
        profile.style.opacity = '1';
        profile.style.transform = 'translateY(0)';
    }, 300);
}

// Handle scroll events to change developer profile
function handleScroll(event) {
    event.preventDefault();
    if (isScrolling) return;
    
    isScrolling = true;
    const delta = event.deltaY;
    
    if (delta > 0) {
        // Scrolling down
        currentDeveloper = (currentDeveloper + 1) % developers.length;
    } else {
        // Scrolling up
        currentDeveloper = (currentDeveloper - 1 + developers.length) % developers.length;
    }
    
    updateDeveloperProfile();
    
    setTimeout(() => {
        isScrolling = false;
    }, 1000); // Increased delay to slow down profile changes
}

document.addEventListener('DOMContentLoaded', () => {
    updateDeveloperProfile();
    
    const teamSection = document.getElementById('team');
    teamSection.addEventListener('wheel', handleScroll, { passive: false });
    
    // Add touch event listener for mobile devices
    let touchStartY = 0;
    teamSection.addEventListener('touchstart', (e) => {
        touchStartY = e.touches[0].clientY;
    }, { passive: false });
    
    teamSection.addEventListener('touchmove', (e) => {
        const touchEndY = e.touches[0].clientY;
        const delta = touchStartY - touchEndY;
        if (Math.abs(delta) > 50) { // Threshold to trigger scroll
            handleScroll({ preventDefault: () => {}, deltaY: delta });
            touchStartY = touchEndY;
        }
    }, { passive: false });
});

// Allow normal behavior of clickable links
document.querySelectorAll('.social-button').forEach(button => {
    button.addEventListener('click', (e) => {
        // Make sure the default behavior is not stopped, to allow clicking
        e.stopPropagation(); // Prevents any bubbling events from stopping the link's default behavior.
    });
});
