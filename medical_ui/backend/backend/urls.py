"""
URL configuration for Medical Auth MVP.

Routes:
- /admin/          → Django admin
- /api/auth/       → Authentication endpoints (signup, login)
- /api/gp/         → GP-only endpoints
- /api/specialist/ → Specialist-only endpoints
- /api/reception/  → Receptionist-only endpoints
"""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    # Django admin
    path('admin/', admin.site.urls),
    
    # Accounts app (auth + role-based endpoints)
    path('api/', include('accounts.urls')),
]

