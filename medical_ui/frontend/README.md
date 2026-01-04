# Medical Portal - Frontend

React-based authentication UI for the Medical Portal.

## Features

- ✅ Login page with JWT authentication
- ✅ Signup page with role-conditional fields
- ✅ Role selection (GP, Specialist, Receptionist)
- ✅ Form validation
- ✅ Clean, professional medical styling
- ✅ Mobile responsive

---

## Quick Start

### 1. Install Node.js

If you don't have Node.js installed:
- **Mac**: `brew install node`
- **Or download**: https://nodejs.org/

### 2. Install Dependencies

```bash
cd /Volumes/1.555212211254./medical_ai/medical_ui/frontend
npm install
```

### 3. Start Development Server

```bash
npm start
```

Frontend runs at: http://localhost:3000

---

## Pages

| Route | Description |
|-------|-------------|
| `/login` | Login page |
| `/signup` | Signup page with role selection |
| `/gp/dashboard` | GP dashboard (placeholder) |
| `/specialist/dashboard` | Specialist dashboard (placeholder) |
| `/reception/dashboard` | Receptionist dashboard (placeholder) |

---

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── index.js          # Entry point
│   ├── App.js            # Routes & main component
│   ├── api.js            # API service (fetch calls)
│   ├── styles.css        # Global styles
│   └── pages/
│       ├── Login.js      # Login page component
│       └── Signup.js     # Signup page component
└── package.json
```

---

## API Integration

The frontend communicates with the Django backend at `http://127.0.0.1:8000/api/`.

### Endpoints Used

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/signup/` | POST | Register new user |
| `/api/auth/login/` | POST | Get JWT tokens |

---

## Running Both Servers

**Terminal 1 - Backend (Django):**
```bash
cd /Volumes/1.555212211254./medical_ai/medical_ui/backend
source /Volumes/1.555212211254./medical_ai/.venv/bin/activate
python manage.py runserver
```

**Terminal 2 - Frontend (React):**
```bash
cd /Volumes/1.555212211254./medical_ai/medical_ui/frontend
npm start
```

---

## Screenshots

### Login Page
- Clean, professional design
- Email & password fields
- Link to signup page

### Signup Page
- Name, email, password fields
- Role dropdown (GP, Specialist, Receptionist)
- Conditional fields based on role:
  - **GP**: License number
  - **Specialist**: Specialty + License number
  - **Receptionist**: Department

