/**
 * Main App Component
 * Routes: /login, /signup, and role-based dashboard redirects
 */

import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Login from './pages/Login';
import Signup from './pages/Signup';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Auth routes */}
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        
        {/* Placeholder dashboard routes (redirect to login for now) */}
        <Route path="/gp/dashboard" element={<DashboardPlaceholder role="GP" />} />
        <Route path="/specialist/dashboard" element={<DashboardPlaceholder role="Specialist" />} />
        <Route path="/reception/dashboard" element={<DashboardPlaceholder role="Receptionist" />} />
        
        {/* Default redirect */}
        <Route path="/" element={<Navigate to="/login" replace />} />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

/**
 * Placeholder component for dashboard routes
 * Shows a simple message until dashboards are implemented
 */
function DashboardPlaceholder({ role }) {
  const handleLogout = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user_role');
    localStorage.removeItem('user_id');
    window.location.href = '/login';
  };

  return (
    <div className="dashboard-placeholder">
      <div className="placeholder-card">
        <div className="placeholder-icon">✓</div>
        <h1>Welcome, {role}!</h1>
        <p>You have successfully logged in.</p>
        <p className="placeholder-note">Dashboard coming soon...</p>
        <button onClick={handleLogout} className="btn btn-secondary">
          Logout
        </button>
      </div>
    </div>
  );
}

export default App;

