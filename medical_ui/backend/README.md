# Medical Auth MVP - Django Backend

A Django REST API for user authentication and role-based access control.

## Features

- ✅ Custom User model with email as username
- ✅ Role-based access (GP, SPECIALIST, RECEPTIONIST, ADMIN)
- ✅ JWT authentication (access + refresh tokens)
- ✅ Role-specific user profiles
- ✅ Role-based permission classes
- ✅ Protected endpoints per role

---

## Quick Start

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Create superuser (optional)
python manage.py createsuperuser

# Run server
python manage.py runserver
```

Server runs at: `http://127.0.0.1:8000`

---

## API Endpoints

### Authentication

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/auth/signup/` | Register new user | ❌ |
| POST | `/api/auth/login/` | Login, get JWT | ❌ |
| GET | `/api/auth/me/` | Get current user | ✅ |
| POST | `/api/auth/token/refresh/` | Refresh JWT token | ❌ (needs refresh token) |

### Role-Protected Endpoints

| Method | Endpoint | Allowed Roles |
|--------|----------|---------------|
| GET | `/api/gp/dashboard/` | GP only |
| GET | `/api/specialist/dashboard/` | SPECIALIST only |
| GET | `/api/reception/dashboard/` | RECEPTIONIST only |

---

## API Examples

### 1. Signup (GP)

```bash
curl -X POST http://127.0.0.1:8000/api/auth/signup/ \
  -H "Content-Type: application/json" \
  -d '{
    "first_name": "John",
    "last_name": "Smith",
    "email": "john.smith@hospital.com",
    "password": "SecurePass123!",
    "role": "GP",
    "organization": "City Hospital",
    "license_number": "MD-12345"
  }'
```

**Response (201):**
```json
{
  "message": "User registered successfully",
  "user": {
    "id": 1,
    "first_name": "John",
    "last_name": "Smith",
    "email": "john.smith@hospital.com",
    "role": "GP",
    "organization": "City Hospital",
    "is_active": true,
    "created_at": "2025-01-01T10:00:00Z",
    "profile": {
      "specialty": null,
      "department": null,
      "license_number": "MD-12345"
    }
  }
}
```

### 2. Signup (Specialist)

```bash
curl -X POST http://127.0.0.1:8000/api/auth/signup/ \
  -H "Content-Type: application/json" \
  -d '{
    "first_name": "Sarah",
    "last_name": "Johnson",
    "email": "sarah.j@hospital.com",
    "password": "SecurePass123!",
    "role": "SPECIALIST",
    "organization": "City Hospital",
    "specialty": "Cardiology",
    "license_number": "SP-67890"
  }'
```

### 3. Signup (Receptionist)

```bash
curl -X POST http://127.0.0.1:8000/api/auth/signup/ \
  -H "Content-Type: application/json" \
  -d '{
    "first_name": "Emily",
    "last_name": "Davis",
    "email": "emily.d@hospital.com",
    "password": "SecurePass123!",
    "role": "RECEPTIONIST",
    "department": "Emergency"
  }'
```

### 4. Login

```bash
curl -X POST http://127.0.0.1:8000/api/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john.smith@hospital.com",
    "password": "SecurePass123!"
  }'
```

**Response (200):**
```json
{
  "access": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user_id": 1,
  "role": "GP"
}
```

### 5. Access Protected Endpoint

```bash
# GP accessing GP dashboard (✅ allowed)
curl -X GET http://127.0.0.1:8000/api/gp/dashboard/ \
  -H "Authorization: Bearer <access_token>"

# Response (200):
{
  "message": "Welcome to the GP Dashboard",
  "user": "John Smith",
  "role": "GP",
  "description": "This endpoint is only accessible to GPs."
}
```

```bash
# GP accessing Specialist dashboard (❌ denied)
curl -X GET http://127.0.0.1:8000/api/specialist/dashboard/ \
  -H "Authorization: Bearer <access_token>"

# Response (403):
{
  "detail": "Access denied. This endpoint is only available to Specialists."
}
```

### 6. Refresh Token

```bash
curl -X POST http://127.0.0.1:8000/api/auth/token/refresh/ \
  -H "Content-Type: application/json" \
  -d '{
    "refresh": "<refresh_token>"
  }'
```

**Response (200):**
```json
{
  "access": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

---

## User Roles

| Role | Description | Required Fields |
|------|-------------|-----------------|
| `GP` | General Practitioner | `license_number` |
| `SPECIALIST` | Medical Specialist | `specialty`, `license_number` |
| `RECEPTIONIST` | Reception Staff | `department` (optional) |
| `ADMIN` | Administrator | (reserved for superusers) |

---

## Project Structure

```
backend/
├── manage.py
├── requirements.txt
├── db.sqlite3
├── backend/
│   ├── __init__.py
│   ├── settings.py       # Django settings, JWT config
│   ├── urls.py           # Root URL routing
│   └── wsgi.py
└── accounts/
    ├── __init__.py
    ├── admin.py          # Django admin config
    ├── apps.py           # App config
    ├── models.py         # User, UserProfile models
    ├── serializers.py    # DRF serializers
    ├── permissions.py    # Role-based permissions
    ├── views.py          # API views
    ├── urls.py           # App URL routing
    └── migrations/
```

---

## Security Notes

- Passwords are hashed using Django's PBKDF2 (default)
- JWT tokens expire after 1 hour (configurable in settings)
- Refresh tokens expire after 7 days
- Roles cannot be changed via API after signup
- Role validation on every protected endpoint

---

## Admin Access

```
URL: http://127.0.0.1:8000/admin/
Default: admin@medical.com / admin123
```

---

## Next Steps

1. Patient model
2. GP dashboard logic
3. Referral document model
4. Switch to PostgreSQL for production
5. Add CORS configuration for frontend

