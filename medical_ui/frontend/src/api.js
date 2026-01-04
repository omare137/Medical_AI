/**
 * API Service for Medical Auth
 * Handles all communication with Django backend
 */

// Backend API base URL
const API_BASE = 'http://127.0.0.1:8000/api';

/**
 * Generic fetch wrapper with error handling
 */
async function apiRequest(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  
  const defaultHeaders = {
    'Content-Type': 'application/json',
  };
  
  // Add auth token if available
  const token = localStorage.getItem('access_token');
  if (token) {
    defaultHeaders['Authorization'] = `Bearer ${token}`;
  }
  
  const config = {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers,
    },
  };
  
  const response = await fetch(url, config);
  const data = await response.json();
  
  if (!response.ok) {
    // Extract error message from Django response
    const errorMessage = data.errors 
      ? Object.values(data.errors).flat().join(', ')
      : data.detail || 'An error occurred';
    throw new Error(errorMessage);
  }
  
  return data;
}

/**
 * Sign up a new user
 * @param {Object} userData - User registration data
 */
export async function signup(userData) {
  return apiRequest('/auth/signup/', {
    method: 'POST',
    body: JSON.stringify(userData),
  });
}

/**
 * Log in user and get JWT tokens
 * @param {string} email - User email
 * @param {string} password - User password
 */
export async function login(email, password) {
  const data = await apiRequest('/auth/login/', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
  
  // Store tokens and user info
  localStorage.setItem('access_token', data.access);
  localStorage.setItem('refresh_token', data.refresh);
  localStorage.setItem('user_role', data.role);
  localStorage.setItem('user_id', data.user_id);
  
  return data;
}

/**
 * Get dashboard redirect path based on role
 * @param {string} role - User role
 */
export function getDashboardPath(role) {
  switch (role) {
    case 'GP':
      return '/gp/dashboard';
    case 'SPECIALIST':
      return '/specialist/dashboard';
    case 'RECEPTIONIST':
      return '/reception/dashboard';
    default:
      return '/login';
  }
}

/**
 * Check if user is logged in
 */
export function isAuthenticated() {
  return !!localStorage.getItem('access_token');
}

/**
 * Get current user role
 */
export function getUserRole() {
  return localStorage.getItem('user_role');
}

