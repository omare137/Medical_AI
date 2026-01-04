"""
URL patterns for accounts app.

Routes:
- /api/auth/signup/          → POST - Register new user
- /api/auth/login/           → POST - Login and get JWT
- /api/auth/me/              → GET  - Get current user
- /api/auth/token/refresh/   → POST - Refresh JWT token
- /api/gp/dashboard/         → GET  - GP-only endpoint
- /api/specialist/dashboard/ → GET  - Specialist-only endpoint
- /api/reception/dashboard/  → GET  - Receptionist-only endpoint
"""

from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from .views import (
    SignupView,
    LoginView,
    MeView,
    GPDashboardView,
    SpecialistDashboardView,
    ReceptionDashboardView,
)

urlpatterns = [
    # ==========================================================================
    # Authentication endpoints
    # ==========================================================================
    path('auth/signup/', SignupView.as_view(), name='signup'),
    path('auth/login/', LoginView.as_view(), name='login'),
    path('auth/me/', MeView.as_view(), name='me'),
    
    # JWT token refresh (using SimpleJWT's built-in view)
    path('auth/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    
    # ==========================================================================
    # Role-protected endpoints (test/placeholder views)
    # ==========================================================================
    path('gp/dashboard/', GPDashboardView.as_view(), name='gp_dashboard'),
    path('specialist/dashboard/', SpecialistDashboardView.as_view(), name='specialist_dashboard'),
    path('reception/dashboard/', ReceptionDashboardView.as_view(), name='reception_dashboard'),
]

