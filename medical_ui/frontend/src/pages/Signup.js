/**
 * Signup Page Component
 * Handles user registration with role-conditional fields
 */

import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { signup } from '../api';

// Available roles for signup
const ROLES = [
  { value: '', label: 'Select your role...' },
  { value: 'GP', label: 'General Practitioner (GP)' },
  { value: 'SPECIALIST', label: 'Specialist' },
  { value: 'RECEPTIONIST', label: 'Receptionist' },
];

function Signup() {
  const navigate = useNavigate();
  
  // Form state
  const [formData, setFormData] = useState({
    first_name: '',
    last_name: '',
    email: '',
    password: '',
    confirmPassword: '',
    role: '',
    organization: '',
    // Role-specific fields
    specialty: '',
    license_number: '',
    department: '',
  });
  
  // UI state
  const [error, setError] = useState('');
  const [fieldErrors, setFieldErrors] = useState({});
  const [loading, setLoading] = useState(false);
  
  /**
   * Handle input changes
   */
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    
    // Clear field error when user types
    if (fieldErrors[name]) {
      setFieldErrors(prev => ({ ...prev, [name]: '' }));
    }
  };
  
  /**
   * Validate form before submission
   */
  const validateForm = () => {
    const errors = {};
    
    if (!formData.first_name.trim()) {
      errors.first_name = 'First name is required';
    }
    
    if (!formData.last_name.trim()) {
      errors.last_name = 'Last name is required';
    }
    
    if (!formData.email.trim()) {
      errors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      errors.email = 'Please enter a valid email';
    }
    
    if (!formData.password) {
      errors.password = 'Password is required';
    } else if (formData.password.length < 8) {
      errors.password = 'Password must be at least 8 characters';
    }
    
    if (formData.password !== formData.confirmPassword) {
      errors.confirmPassword = 'Passwords do not match';
    }
    
    if (!formData.role) {
      errors.role = 'Please select a role';
    }
    
    // Role-specific validation
    if (formData.role === 'GP') {
      if (!formData.license_number.trim()) {
        errors.license_number = 'License number is required for GPs';
      }
    }
    
    if (formData.role === 'SPECIALIST') {
      if (!formData.specialty.trim()) {
        errors.specialty = 'Specialty is required for Specialists';
      }
      if (!formData.license_number.trim()) {
        errors.license_number = 'License number is required for Specialists';
      }
    }
    
    setFieldErrors(errors);
    return Object.keys(errors).length === 0;
  };
  
  /**
   * Handle form submission
   */
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    if (!validateForm()) {
      return;
    }
    
    setLoading(true);
    
    try {
      // Build payload based on role
      const payload = {
        first_name: formData.first_name,
        last_name: formData.last_name,
        email: formData.email,
        password: formData.password,
        role: formData.role,
      };
      
      // Add optional fields
      if (formData.organization.trim()) {
        payload.organization = formData.organization;
      }
      
      // Add role-specific fields
      if (formData.role === 'GP' || formData.role === 'SPECIALIST') {
        payload.license_number = formData.license_number;
      }
      
      if (formData.role === 'SPECIALIST' || formData.role === 'GP') {
        if (formData.specialty.trim()) {
          payload.specialty = formData.specialty;
        }
      }
      
      if (formData.role === 'RECEPTIONIST' && formData.department.trim()) {
        payload.department = formData.department;
      }
      
      await signup(payload);
      
      // Redirect to login with success message
      navigate('/login', {
        state: { message: 'Account created successfully! Please sign in.' }
      });
      
    } catch (err) {
      setError(err.message || 'Failed to create account');
    } finally {
      setLoading(false);
    }
  };
  
  /**
   * Render role-specific fields based on selected role
   */
  const renderRoleFields = () => {
    switch (formData.role) {
      case 'GP':
        return (
          <>
            <div className="form-group">
              <label htmlFor="specialty">Specialty (Optional)</label>
              <input
                type="text"
                id="specialty"
                name="specialty"
                value={formData.specialty}
                onChange={handleChange}
                placeholder="e.g., Family Medicine"
                disabled={loading}
              />
            </div>
            <div className="form-group">
              <label htmlFor="license_number">License Number *</label>
              <input
                type="text"
                id="license_number"
                name="license_number"
                value={formData.license_number}
                onChange={handleChange}
                placeholder="e.g., MD-12345"
                disabled={loading}
                className={fieldErrors.license_number ? 'input-error' : ''}
              />
              {fieldErrors.license_number && (
                <span className="field-error">{fieldErrors.license_number}</span>
              )}
            </div>
          </>
        );
        
      case 'SPECIALIST':
        return (
          <>
            <div className="form-group">
              <label htmlFor="specialty">Specialty *</label>
              <input
                type="text"
                id="specialty"
                name="specialty"
                value={formData.specialty}
                onChange={handleChange}
                placeholder="e.g., Cardiology, Neurology"
                disabled={loading}
                className={fieldErrors.specialty ? 'input-error' : ''}
              />
              {fieldErrors.specialty && (
                <span className="field-error">{fieldErrors.specialty}</span>
              )}
            </div>
            <div className="form-group">
              <label htmlFor="license_number">License Number *</label>
              <input
                type="text"
                id="license_number"
                name="license_number"
                value={formData.license_number}
                onChange={handleChange}
                placeholder="e.g., SP-67890"
                disabled={loading}
                className={fieldErrors.license_number ? 'input-error' : ''}
              />
              {fieldErrors.license_number && (
                <span className="field-error">{fieldErrors.license_number}</span>
              )}
            </div>
          </>
        );
        
      case 'RECEPTIONIST':
        return (
          <div className="form-group">
            <label htmlFor="department">Department (Optional)</label>
            <input
              type="text"
              id="department"
              name="department"
              value={formData.department}
              onChange={handleChange}
              placeholder="e.g., Emergency, Outpatient"
              disabled={loading}
            />
          </div>
        );
        
      default:
        return null;
    }
  };
  
  return (
    <div className="auth-container">
      <div className="auth-card auth-card-wide">
        {/* Header */}
        <div className="auth-header">
          <div className="auth-logo">
            <span className="logo-icon">+</span>
          </div>
          <h1>Create Account</h1>
          <p>Join the Medical Portal</p>
        </div>
        
        {/* Error Message */}
        {error && (
          <div className="alert alert-error">
            {error}
          </div>
        )}
        
        {/* Signup Form */}
        <form onSubmit={handleSubmit} className="auth-form">
          {/* Name Fields */}
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="first_name">First Name *</label>
              <input
                type="text"
                id="first_name"
                name="first_name"
                value={formData.first_name}
                onChange={handleChange}
                placeholder="John"
                disabled={loading}
                className={fieldErrors.first_name ? 'input-error' : ''}
              />
              {fieldErrors.first_name && (
                <span className="field-error">{fieldErrors.first_name}</span>
              )}
            </div>
            
            <div className="form-group">
              <label htmlFor="last_name">Last Name *</label>
              <input
                type="text"
                id="last_name"
                name="last_name"
                value={formData.last_name}
                onChange={handleChange}
                placeholder="Doe"
                disabled={loading}
                className={fieldErrors.last_name ? 'input-error' : ''}
              />
              {fieldErrors.last_name && (
                <span className="field-error">{fieldErrors.last_name}</span>
              )}
            </div>
          </div>
          
          {/* Email */}
          <div className="form-group">
            <label htmlFor="email">Email Address *</label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              placeholder="you@hospital.com"
              autoComplete="email"
              disabled={loading}
              className={fieldErrors.email ? 'input-error' : ''}
            />
            {fieldErrors.email && (
              <span className="field-error">{fieldErrors.email}</span>
            )}
          </div>
          
          {/* Password Fields */}
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="password">Password *</label>
              <input
                type="password"
                id="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="Min. 8 characters"
                autoComplete="new-password"
                disabled={loading}
                className={fieldErrors.password ? 'input-error' : ''}
              />
              {fieldErrors.password && (
                <span className="field-error">{fieldErrors.password}</span>
              )}
            </div>
            
            <div className="form-group">
              <label htmlFor="confirmPassword">Confirm Password *</label>
              <input
                type="password"
                id="confirmPassword"
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleChange}
                placeholder="Re-enter password"
                autoComplete="new-password"
                disabled={loading}
                className={fieldErrors.confirmPassword ? 'input-error' : ''}
              />
              {fieldErrors.confirmPassword && (
                <span className="field-error">{fieldErrors.confirmPassword}</span>
              )}
            </div>
          </div>
          
          {/* Role Selection */}
          <div className="form-group">
            <label htmlFor="role">Role *</label>
            <select
              id="role"
              name="role"
              value={formData.role}
              onChange={handleChange}
              disabled={loading}
              className={fieldErrors.role ? 'input-error' : ''}
            >
              {ROLES.map(role => (
                <option key={role.value} value={role.value}>
                  {role.label}
                </option>
              ))}
            </select>
            {fieldErrors.role && (
              <span className="field-error">{fieldErrors.role}</span>
            )}
          </div>
          
          {/* Organization (Optional) */}
          <div className="form-group">
            <label htmlFor="organization">Organization / Clinic (Optional)</label>
            <input
              type="text"
              id="organization"
              name="organization"
              value={formData.organization}
              onChange={handleChange}
              placeholder="e.g., City General Hospital"
              disabled={loading}
            />
          </div>
          
          {/* Role-Specific Fields */}
          {formData.role && (
            <div className="role-fields">
              <div className="role-fields-header">
                <span className="role-badge">{formData.role}</span>
                <span>Additional Information</span>
              </div>
              {renderRoleFields()}
            </div>
          )}
          
          <button 
            type="submit" 
            className="btn btn-primary btn-full"
            disabled={loading}
          >
            {loading ? 'Creating Account...' : 'Create Account'}
          </button>
        </form>
        
        {/* Footer */}
        <div className="auth-footer">
          <p>
            Already have an account?{' '}
            <Link to="/login">Sign in</Link>
          </p>
        </div>
      </div>
    </div>
  );
}

export default Signup;

