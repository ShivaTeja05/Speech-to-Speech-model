/**
 * Apollo Hospital Admin Panel JavaScript
 * Handles all CRUD operations for settings, doctors, departments, and FAQs
 */

// ============================================================================
// STATE
// ============================================================================

const state = {
    config: {},
    doctors: [],
    departments: [],
    faqs: []
};

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    loadAllData();
    setupFormHandlers();
});

function initTabs() {
    const navItems = document.querySelectorAll('.nav-item');
    
    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const tabId = item.dataset.tab;
            
            // Update nav
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            
            // Update content
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
        });
    });
}

async function loadAllData() {
    await Promise.all([
        loadSettings(),
        loadDoctors(),
        loadDepartments(),
        loadFaqs()
    ]);
}

function setupFormHandlers() {
    // Settings form
    document.getElementById('settingsForm').addEventListener('submit', saveSettings);
    
    // Escalation form
    document.getElementById('escalationForm').addEventListener('submit', saveEscalation);
    
    // Doctor form
    document.getElementById('doctorForm').addEventListener('submit', saveDoctor);
    
    // Department form
    document.getElementById('departmentForm').addEventListener('submit', saveDepartment);
    
    // FAQ form
    document.getElementById('faqForm').addEventListener('submit', saveFaq);
}

// ============================================================================
// SETTINGS
// ============================================================================

async function loadSettings() {
    try {
        const response = await fetch('/api/admin/config');
        if (response.ok) {
            state.config = await response.json();
            populateSettingsForm(state.config);
        }
    } catch (error) {
        console.error('Error loading settings:', error);
        showToast('Failed to load settings', 'error');
    }
}

function populateSettingsForm(config) {
    // Text fields
    document.getElementById('hospitalName').value = config.hospital_name || 'Apollo Hospital';
    document.getElementById('city').value = config.city || 'Bengaluru';
    document.getElementById('emergencyNumber').value = config.emergency_number || '108';
    document.getElementById('helpline').value = config.helpline || '1860-500-1066';
    document.getElementById('maxWords').value = config.max_words || 50;
    document.getElementById('sessionTimeout').value = config.session_timeout_minutes || 30;
    document.getElementById('disclaimerText').value = config.disclaimer_text || '';
    
    // Selects
    document.getElementById('tone').value = config.tone || 'formal';
    document.getElementById('primaryLanguage').value = config.primary_language || 'en';
    
    // Checkboxes
    document.getElementById('disclaimerRequired').checked = config.disclaimer_required !== false;
    document.getElementById('alwaysRecommendDoctor').checked = config.always_recommend_doctor !== false;
    document.getElementById('ttsEnabled').checked = config.tts_enabled !== false;
    
    // Escalation keywords
    const keywords = config.escalation_keywords || [];
    document.getElementById('escalationKeywords').value = keywords.join('\n');
}

async function saveSettings(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const config = {
        hospital_name: formData.get('hospital_name'),
        city: formData.get('city'),
        emergency_number: formData.get('emergency_number'),
        helpline: formData.get('helpline'),
        tone: formData.get('tone'),
        max_words: parseInt(formData.get('max_words')),
        primary_language: formData.get('primary_language'),
        session_timeout_minutes: parseInt(formData.get('session_timeout_minutes')),
        disclaimer_required: formData.get('disclaimer_required') === 'on',
        always_recommend_doctor: formData.get('always_recommend_doctor') === 'on',
        tts_enabled: formData.get('tts_enabled') === 'on',
        disclaimer_text: formData.get('disclaimer_text'),
        supported_languages: ['kn', 'ta', 'te', 'hi', 'en']
    };
    
    try {
        const response = await fetch('/api/admin/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        if (response.ok) {
            showToast('Settings saved successfully');
            state.config = config;
        } else {
            throw new Error('Failed to save');
        }
    } catch (error) {
        console.error('Error saving settings:', error);
        showToast('Failed to save settings', 'error');
    }
}

async function saveEscalation(e) {
    e.preventDefault();
    
    const keywordsText = document.getElementById('escalationKeywords').value;
    const keywords = keywordsText.split('\n').map(k => k.trim()).filter(k => k);
    
    try {
        const response = await fetch('/api/admin/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...state.config,
                escalation_keywords: keywords
            })
        });
        
        if (response.ok) {
            showToast('Escalation rules saved successfully');
            state.config.escalation_keywords = keywords;
        } else {
            throw new Error('Failed to save');
        }
    } catch (error) {
        console.error('Error saving escalation rules:', error);
        showToast('Failed to save escalation rules', 'error');
    }
}

// ============================================================================
// DOCTORS
// ============================================================================

async function loadDoctors() {
    try {
        const response = await fetch('/api/admin/doctors');
        if (response.ok) {
            state.doctors = await response.json();
            renderDoctorsTable();
        }
    } catch (error) {
        console.error('Error loading doctors:', error);
    }
}

function renderDoctorsTable() {
    const tbody = document.querySelector('#doctorsTable tbody');
    const emptyState = document.getElementById('doctorsEmpty');
    
    if (state.doctors.length === 0) {
        tbody.innerHTML = '';
        emptyState.style.display = 'block';
        return;
    }
    
    emptyState.style.display = 'none';
    tbody.innerHTML = state.doctors.map(doc => `
        <tr>
            <td>${doc.name}${doc.name_kn ? ` <span class="lang-badge">${doc.name_kn}</span>` : ''}</td>
            <td>${doc.specialization || '-'}</td>
            <td>${doc.department || '-'}</td>
            <td>${Array.isArray(doc.available_days) ? doc.available_days.join(', ') : doc.available_days || '-'}</td>
            <td>${doc.timings || '-'}</td>
            <td class="actions">
                <button class="btn-icon" onclick="editDoctor('${doc.id}')" title="Edit">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                    </svg>
                </button>
                <button class="btn-icon btn-danger" onclick="deleteDoctor('${doc.id}')" title="Delete">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"/>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                    </svg>
                </button>
            </td>
        </tr>
    `).join('');
}

function showDoctorModal(doctor = null) {
    const modal = document.getElementById('doctorModal');
    const form = document.getElementById('doctorForm');
    const title = document.getElementById('doctorModalTitle');
    
    form.reset();
    
    if (doctor) {
        title.textContent = 'Edit Doctor';
        document.getElementById('doctorId').value = doctor.id;
        document.getElementById('doctorName').value = doctor.name || '';
        document.getElementById('doctorNameKn').value = doctor.name_kn || '';
        document.getElementById('doctorSpec').value = doctor.specialization || '';
        document.getElementById('doctorDept').value = doctor.department || '';
        document.getElementById('doctorTimings').value = doctor.timings || '';
        
        // Set available days checkboxes
        const days = Array.isArray(doctor.available_days) ? doctor.available_days : [];
        document.querySelectorAll('input[name="available_days"]').forEach(cb => {
            cb.checked = days.includes(cb.value);
        });
    } else {
        title.textContent = 'Add Doctor';
        document.getElementById('doctorId').value = '';
    }
    
    modal.classList.add('active');
}

function editDoctor(id) {
    const doctor = state.doctors.find(d => d.id === id);
    if (doctor) {
        showDoctorModal(doctor);
    }
}

async function saveDoctor(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const availableDays = [];
    document.querySelectorAll('input[name="available_days"]:checked').forEach(cb => {
        availableDays.push(cb.value);
    });
    
    const doctor = {
        name: formData.get('name'),
        name_kn: formData.get('name_kn'),
        specialization: formData.get('specialization'),
        department: formData.get('department'),
        timings: formData.get('timings'),
        available_days: availableDays
    };
    
    const id = formData.get('id');
    
    try {
        let response;
        if (id) {
            response = await fetch(`/api/admin/doctors/${id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(doctor)
            });
        } else {
            response = await fetch('/api/admin/doctors', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(doctor)
            });
        }
        
        if (response.ok) {
            showToast(id ? 'Doctor updated' : 'Doctor added');
            closeModal('doctorModal');
            await loadDoctors();
        } else {
            throw new Error('Failed to save');
        }
    } catch (error) {
        console.error('Error saving doctor:', error);
        showToast('Failed to save doctor', 'error');
    }
}

async function deleteDoctor(id) {
    if (!confirm('Are you sure you want to delete this doctor?')) return;
    
    try {
        const response = await fetch(`/api/admin/doctors/${id}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            showToast('Doctor deleted');
            await loadDoctors();
        } else {
            throw new Error('Failed to delete');
        }
    } catch (error) {
        console.error('Error deleting doctor:', error);
        showToast('Failed to delete doctor', 'error');
    }
}

// ============================================================================
// DEPARTMENTS
// ============================================================================

async function loadDepartments() {
    try {
        const response = await fetch('/api/admin/departments');
        if (response.ok) {
            state.departments = await response.json();
            renderDepartmentsTable();
        }
    } catch (error) {
        console.error('Error loading departments:', error);
    }
}

function renderDepartmentsTable() {
    const tbody = document.querySelector('#departmentsTable tbody');
    const emptyState = document.getElementById('departmentsEmpty');
    
    if (state.departments.length === 0) {
        tbody.innerHTML = '';
        emptyState.style.display = 'block';
        return;
    }
    
    emptyState.style.display = 'none';
    tbody.innerHTML = state.departments.map(dept => `
        <tr>
            <td>${dept.name}</td>
            <td>${dept.floor || '-'}</td>
            <td>${dept.timings || '-'}</td>
            <td>${dept.contact || '-'}</td>
            <td class="actions">
                <button class="btn-icon btn-danger" onclick="deleteDepartment('${dept.id}')" title="Delete">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"/>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                    </svg>
                </button>
            </td>
        </tr>
    `).join('');
}

function showDepartmentModal() {
    const modal = document.getElementById('departmentModal');
    const form = document.getElementById('departmentForm');
    
    form.reset();
    document.getElementById('departmentId').value = '';
    modal.classList.add('active');
}

async function saveDepartment(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const department = {
        name: formData.get('name'),
        floor: formData.get('floor'),
        timings: formData.get('timings'),
        contact: formData.get('contact')
    };
    
    try {
        const response = await fetch('/api/admin/departments', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(department)
        });
        
        if (response.ok) {
            showToast('Department added');
            closeModal('departmentModal');
            await loadDepartments();
        } else {
            throw new Error('Failed to save');
        }
    } catch (error) {
        console.error('Error saving department:', error);
        showToast('Failed to save department', 'error');
    }
}

async function deleteDepartment(id) {
    if (!confirm('Are you sure you want to delete this department?')) return;
    
    try {
        const response = await fetch(`/api/admin/departments/${id}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            showToast('Department deleted');
            await loadDepartments();
        } else {
            throw new Error('Failed to delete');
        }
    } catch (error) {
        console.error('Error deleting department:', error);
        showToast('Failed to delete department', 'error');
    }
}

// ============================================================================
// FAQS
// ============================================================================

async function loadFaqs() {
    try {
        const response = await fetch('/api/admin/faqs');
        if (response.ok) {
            state.faqs = await response.json();
            renderFaqsTable();
        }
    } catch (error) {
        console.error('Error loading FAQs:', error);
    }
}

function renderFaqsTable() {
    const tbody = document.querySelector('#faqsTable tbody');
    const emptyState = document.getElementById('faqsEmpty');
    
    if (state.faqs.length === 0) {
        tbody.innerHTML = '';
        emptyState.style.display = 'block';
        return;
    }
    
    emptyState.style.display = 'none';
    tbody.innerHTML = state.faqs.map(faq => `
        <tr>
            <td>${truncate(faq.question, 50)}</td>
            <td>${truncate(faq.answer, 80)}</td>
            <td><span class="category-badge">${faq.category || 'general'}</span></td>
            <td class="actions">
                <button class="btn-icon" onclick="editFaq('${faq.id}')" title="Edit">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                    </svg>
                </button>
                <button class="btn-icon btn-danger" onclick="deleteFaq('${faq.id}')" title="Delete">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"/>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                    </svg>
                </button>
            </td>
        </tr>
    `).join('');
}

function showFaqModal(faq = null) {
    const modal = document.getElementById('faqModal');
    const form = document.getElementById('faqForm');
    const title = document.getElementById('faqModalTitle');
    
    form.reset();
    
    if (faq) {
        title.textContent = 'Edit FAQ';
        document.getElementById('faqId').value = faq.id;
        document.getElementById('faqQuestion').value = faq.question || '';
        document.getElementById('faqQuestionKn').value = faq.question_kn || '';
        document.getElementById('faqAnswer').value = faq.answer || '';
        document.getElementById('faqAnswerKn').value = faq.answer_kn || '';
        document.getElementById('faqCategory').value = faq.category || 'general';
    } else {
        title.textContent = 'Add FAQ';
        document.getElementById('faqId').value = '';
    }
    
    modal.classList.add('active');
}

function editFaq(id) {
    const faq = state.faqs.find(f => f.id === id);
    if (faq) {
        showFaqModal(faq);
    }
}

async function saveFaq(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const faq = {
        question: formData.get('question'),
        question_kn: formData.get('question_kn'),
        answer: formData.get('answer'),
        answer_kn: formData.get('answer_kn'),
        category: formData.get('category')
    };
    
    const id = formData.get('id');
    
    try {
        let response;
        if (id) {
            response = await fetch(`/api/admin/faqs/${id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(faq)
            });
        } else {
            response = await fetch('/api/admin/faqs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(faq)
            });
        }
        
        if (response.ok) {
            showToast(id ? 'FAQ updated' : 'FAQ added');
            closeModal('faqModal');
            await loadFaqs();
        } else {
            throw new Error('Failed to save');
        }
    } catch (error) {
        console.error('Error saving FAQ:', error);
        showToast('Failed to save FAQ', 'error');
    }
}

async function deleteFaq(id) {
    if (!confirm('Are you sure you want to delete this FAQ?')) return;
    
    try {
        const response = await fetch(`/api/admin/faqs/${id}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            showToast('FAQ deleted');
            await loadFaqs();
        } else {
            throw new Error('Failed to delete');
        }
    } catch (error) {
        console.error('Error deleting FAQ:', error);
        showToast('Failed to delete FAQ', 'error');
    }
}

// ============================================================================
// UTILITIES
// ============================================================================

function closeModal(modalId) {
    document.getElementById(modalId).classList.remove('active');
}

function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function truncate(str, len) {
    if (!str) return '-';
    return str.length > len ? str.substring(0, len) + '...' : str;
}

// Close modal when clicking outside
document.querySelectorAll('.modal').forEach(modal => {
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('active');
        }
    });
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal.active').forEach(m => m.classList.remove('active'));
    }
});
