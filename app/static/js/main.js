// HR ATS JavaScript functionality

// Upload zone functionality
document.addEventListener('DOMContentLoaded', function() {
    initializeUploadZone();
    initializeKanban();
    initializeDebiasingToggle();
    initializePopovers();
});

function initializeUploadZone() {
    const uploadZone = document.getElementById('upload-zone');
    if (!uploadZone) return;

    uploadZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            uploadFiles(files);
        }
    });

    uploadZone.addEventListener('click', function() {
        document.getElementById('file-input').click();
    });

    document.getElementById('file-input').addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            uploadFiles(e.target.files);
        }
    });
}

function uploadFiles(files) {
    const formData = new FormData();
    
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    // Show loading state
    showUploadProgress(true);

    fetch('/api/ingest/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        showUploadProgress(false);
        if (data.success) {
            showNotification('Files uploaded successfully', 'success');
            // Refresh page or update UI
            setTimeout(() => location.reload(), 2000);
        } else {
            showNotification('Upload failed: ' + data.error, 'error');
        }
    })
    .catch(error => {
        showUploadProgress(false);
        showNotification('Upload failed: ' + error.message, 'error');
    });
}

function showUploadProgress(show) {
    const spinner = document.querySelector('.upload-spinner');
    const text = document.querySelector('.upload-text');
    
    if (show) {
        spinner.style.display = 'inline-block';
        text.textContent = 'Uploading and processing files...';
    } else {
        spinner.style.display = 'none';
        text.textContent = 'Drop files here or click to upload';
    }
}

function initializeKanban() {
    const kanbanItems = document.querySelectorAll('.kanban-item');
    
    kanbanItems.forEach(item => {
        item.draggable = true;
        
        item.addEventListener('dragstart', function(e) {
            e.dataTransfer.setData('text/plain', item.dataset.applicationId);
        });
    });

    const kanbanColumns = document.querySelectorAll('.kanban-column');
    
    kanbanColumns.forEach(column => {
        column.addEventListener('dragover', function(e) {
            e.preventDefault();
        });

        column.addEventListener('drop', function(e) {
            e.preventDefault();
            const applicationId = e.dataTransfer.getData('text/plain');
            const newStage = column.dataset.stage;
            
            moveApplication(applicationId, newStage);
        });
    });
}

function moveApplication(applicationId, newStage) {
    fetch(`/api/applications/${applicationId}/move`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ stage: newStage })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Move the item in the UI
            const item = document.querySelector(`[data-application-id="${applicationId}"]`);
            const targetColumn = document.querySelector(`[data-stage="${newStage}"] .kanban-items`);
            targetColumn.appendChild(item);
            
            showNotification('Application moved successfully', 'success');
        } else {
            showNotification('Failed to move application', 'error');
        }
    })
    .catch(error => {
        showNotification('Error: ' + error.message, 'error');
    });
}

function initializeDebiasingToggle() {
    const debiasToggle = document.getElementById('debias-toggle');
    if (!debiasToggle) return;

    debiasToggle.addEventListener('click', function() {
        const isActive = debiasToggle.classList.contains('debias-active');
        
        if (isActive) {
            debiasToggle.classList.remove('debias-active');
            debiasToggle.textContent = 'Enable Debiasing';
            showPersonalInfo(true);
        } else {
            debiasToggle.classList.add('debias-active');
            debiasToggle.textContent = 'Disable Debiasing';
            showPersonalInfo(false);
        }
    });
}

function showPersonalInfo(show) {
    const personalCells = document.querySelectorAll('.personal-info');
    
    personalCells.forEach(cell => {
        if (show) {
            cell.style.display = '';
        } else {
            cell.style.display = 'none';
        }
    });
}

function initializePopovers() {
    // Initialize Bootstrap popovers for explanations
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

function advanceApplication(applicationId) {
    updateApplicationDecision(applicationId, 'advance');
}

function rejectApplication(applicationId) {
    updateApplicationDecision(applicationId, 'reject');
}

function updateApplicationDecision(applicationId, decision) {
    fetch(`/api/applications/${applicationId}/${decision}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification(`Application ${decision}d successfully`, 'success');
            // Update UI or refresh
            location.reload();
        } else {
            showNotification(`Failed to ${decision} application`, 'error');
        }
    })
    .catch(error => {
        showNotification('Error: ' + error.message, 'error');
    });
}

function triggerRanking(roleId) {
    showNotification('Starting ranking process...', 'info');
    
    fetch(`/api/rank/role/${roleId}`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('Ranking completed successfully', 'success');
            setTimeout(() => location.reload(), 2000);
        } else {
            showNotification('Ranking failed: ' + data.error, 'error');
        }
    })
    .catch(error => {
        showNotification('Error: ' + error.message, 'error');
    });
}

function showNotification(message, type) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 5000);
}

// Export functions for global use
window.advanceApplication = advanceApplication;
window.rejectApplication = rejectApplication;
window.triggerRanking = triggerRanking;
