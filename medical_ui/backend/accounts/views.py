"""
Views for User authentication and role-based endpoints.

Endpoints:
- POST /api/auth/signup/     → Register new user
- POST /api/auth/login/      → Login and get JWT tokens
- GET  /api/gp/dashboard/    → GP-only endpoint
- GET  /api/specialist/dashboard/ → Specialist-only endpoint
- GET  /api/reception/dashboard/  → Receptionist-only endpoint
"""

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken

from .models import User
from .serializers import SignupSerializer, LoginSerializer, UserSerializer
from .permissions import IsGP, IsSpecialist, IsReceptionist


# =============================================================================
# AUTHENTICATION VIEWS
# =============================================================================

class SignupView(APIView):
    """
    POST /api/auth/signup/
    
    Register a new user with role and profile.
    
    Request body:
    {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john@example.com",
        "password": "securepassword123",
        "role": "GP",  // GP | SPECIALIST | RECEPTIONIST
        "organization": "City Hospital",  // optional
        "specialty": "Cardiology",  // required for SPECIALIST
        "license_number": "MD12345"  // required for GP and SPECIALIST
    }
    
    Response (201 Created):
    {
        "message": "User registered successfully",
        "user": { ... }
    }
    """
    permission_classes = [AllowAny]  # No auth required for signup
    
    def post(self, request):
        serializer = SignupSerializer(data=request.data)
        
        if serializer.is_valid():
            user = serializer.save()
            
            return Response(
                {
                    'message': 'User registered successfully',
                    'user': UserSerializer(user).data
                },
                status=status.HTTP_201_CREATED
            )
        
        return Response(
            {'errors': serializer.errors},
            status=status.HTTP_400_BAD_REQUEST
        )


class LoginView(APIView):
    """
    POST /api/auth/login/
    
    Authenticate user and return JWT tokens.
    
    Request body:
    {
        "email": "john@example.com",
        "password": "securepassword123"
    }
    
    Response (200 OK):
    {
        "access": "eyJ...",
        "refresh": "eyJ...",
        "user_id": 1,
        "role": "GP"
    }
    """
    permission_classes = [AllowAny]  # No auth required for login
    
    def post(self, request):
        serializer = LoginSerializer(data=request.data, context={'request': request})
        
        if serializer.is_valid():
            user = serializer.validated_data['user']
            
            # Generate JWT tokens
            refresh = RefreshToken.for_user(user)
            access = refresh.access_token
            
            return Response(
                {
                    'access': str(access),
                    'refresh': str(refresh),
                    'user_id': user.id,
                    'role': user.role,
                },
                status=status.HTTP_200_OK
            )
        
        return Response(
            {'errors': serializer.errors},
            status=status.HTTP_401_UNAUTHORIZED
        )


class MeView(APIView):
    """
    GET /api/auth/me/
    
    Get current authenticated user's details.
    Requires valid JWT token.
    
    Response (200 OK):
    {
        "id": 1,
        "first_name": "John",
        "last_name": "Doe",
        "email": "john@example.com",
        "role": "GP",
        ...
    }
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        serializer = UserSerializer(request.user)
        return Response(serializer.data)


# =============================================================================
# ROLE-PROTECTED VIEWS (Test endpoints)
# =============================================================================

class GPDashboardView(APIView):
    """
    GET /api/gp/dashboard/
    
    GP-only endpoint.
    Returns a simple message confirming access.
    
    Permissions: IsAuthenticated + IsGP
    """
    permission_classes = [IsAuthenticated, IsGP]
    
    def get(self, request):
        return Response({
            'message': 'Welcome to the GP Dashboard',
            'user': request.user.get_full_name(),
            'role': request.user.role,
            'description': 'This endpoint is only accessible to GPs.',
        })


class SpecialistDashboardView(APIView):
    """
    GET /api/specialist/dashboard/
    
    Specialist-only endpoint.
    Returns a simple message confirming access.
    
    Permissions: IsAuthenticated + IsSpecialist
    """
    permission_classes = [IsAuthenticated, IsSpecialist]
    
    def get(self, request):
        # Get specialty from profile
        specialty = ''
        if hasattr(request.user, 'profile') and request.user.profile:
            specialty = request.user.profile.specialty
        
        return Response({
            'message': 'Welcome to the Specialist Dashboard',
            'user': request.user.get_full_name(),
            'role': request.user.role,
            'specialty': specialty,
            'description': 'This endpoint is only accessible to Specialists.',
        })


class ReceptionDashboardView(APIView):
    """
    GET /api/reception/dashboard/
    
    Receptionist-only endpoint.
    Returns a simple message confirming access.
    
    Permissions: IsAuthenticated + IsReceptionist
    """
    permission_classes = [IsAuthenticated, IsReceptionist]
    
    def get(self, request):
        # Get department from profile
        department = ''
        if hasattr(request.user, 'profile') and request.user.profile:
            department = request.user.profile.department
        
        return Response({
            'message': 'Welcome to the Reception Dashboard',
            'user': request.user.get_full_name(),
            'role': request.user.role,
            'department': department,
            'description': 'This endpoint is only accessible to Receptionists.',
        })

