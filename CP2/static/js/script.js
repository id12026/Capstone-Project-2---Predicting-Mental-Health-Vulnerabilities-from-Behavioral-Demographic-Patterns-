// Additional JavaScript functionality

// Form validation
function validateForm() {
    const form = document.getElementById('predictionForm');
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            field.classList.add('is-invalid');
            isValid = false;
        } else {
            field.classList.remove('is-invalid');
        }
    });
    
    return isValid;
}

// Initialize tooltips
document.addEventListener('DOMContentLoaded', function() {
    // Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Add input validation
    const form = document.getElementById('predictionForm');
    if (form) {
        const inputs = form.querySelectorAll('input, select');
        inputs.forEach(input => {
            input.addEventListener('change', function() {
                if (this.value.trim()) {
                    this.classList.remove('is-invalid');
                    this.classList.add('is-valid');
                }
            });
        });
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Save assessment result to local storage
function saveAssessmentToLocal(result) {
    try {
        const assessments = JSON.parse(localStorage.getItem('mentalHealthAssessments') || '[]');
        assessments.push({
            timestamp: new Date().toISOString(),
            result: result
        });
        
        // Keep only last 10 assessments
        if (assessments.length > 10) {
            assessments.shift();
        }
        
        localStorage.setItem('mentalHealthAssessments', JSON.stringify(assessments));
        return true;
    } catch (error) {
        console.error('Error saving to local storage:', error);
        return false;
    }
}

// Load assessment history
function loadAssessmentHistory() {
    try {
        const assessments = JSON.parse(localStorage.getItem('mentalHealthAssessments') || '[]');
        return assessments;
    } catch (error) {
        console.error('Error loading from local storage:', error);
        return [];
    }
}

// Export assessment data
function exportAssessmentData() {
    const assessments = loadAssessmentHistory();
    if (assessments.length === 0) {
        alert('No assessment history found.');
        return;
    }
    
    const dataStr = JSON.stringify(assessments, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = 'mental-health-assessments.json';
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
}