"""
Serializers for User authentication and registration.

Features:
- SignupSerializer: Creates user with role and profile
- LoginSerializer: Validates credentials
- UserSerializer: Returns user data with role
"""

from rest_framework import serializers
from django.contrib.auth import authenticate
from django.contrib.auth.password_validation import validate_password
from .models import User, UserProfile, UserRole


# =============================================================================
# USER SERIALIZER (Read-only)
# =============================================================================

class UserProfileSerializer(serializers.ModelSerializer):
    """Serializer for UserProfile (role-specific metadata)."""
    
    class Meta:
        model = UserProfile
        fields = ['specialty', 'department', 'license_number']
        read_only_fields = fields


class UserSerializer(serializers.ModelSerializer):
    """
    Serializer for User model.
    Used for returning user data in responses.
    """
    profile = UserProfileSerializer(read_only=True)
    
    class Meta:
        model = User
        fields = [
            'id',
            'first_name',
            'last_name',
            'email',
            'role',
            'organization',
            'is_active',
            'created_at',
            'profile',
        ]
        read_only_fields = fields


# =============================================================================
# SIGNUP SERIALIZER
# =============================================================================

class SignupSerializer(serializers.Serializer):
    """
    Serializer for user registration.
    
    Accepts:
    - first_name, last_name, email, password
    - role (GP | SPECIALIST | RECEPTIONIST)
    - organization (optional)
    - Role-specific fields: specialty, department, license_number
    
    Behavior:
    - Validates role is one of allowed values
    - Hashes password using Django's password hasher
    - Creates User and related UserProfile
    """
    
    # Required fields
    first_name = serializers.CharField(max_length=150)
    last_name = serializers.CharField(max_length=150)
    email = serializers.EmailField()
    password = serializers.CharField(
        write_only=True,
        min_length=8,
        style={'input_type': 'password'}
    )
    
    # Role - only allow GP, SPECIALIST, RECEPTIONIST for signup
    # ADMIN is reserved for superusers
    role = serializers.ChoiceField(
        choices=[
            (UserRole.GP, 'GP'),
            (UserRole.SPECIALIST, 'SPECIALIST'),
            (UserRole.RECEPTIONIST, 'RECEPTIONIST'),
        ]
    )
    
    # Optional fields
    organization = serializers.CharField(max_length=255, required=False, allow_blank=True)
    
    # Role-specific fields (optional)
    specialty = serializers.CharField(max_length=100, required=False, allow_blank=True)
    department = serializers.CharField(max_length=100, required=False, allow_blank=True)
    license_number = serializers.CharField(max_length=50, required=False, allow_blank=True)
    
    def validate_email(self, value):
        """Ensure email is unique."""
        if User.objects.filter(email__iexact=value).exists():
            raise serializers.ValidationError("A user with this email already exists.")
        return value.lower()
    
    def validate_password(self, value):
        """Validate password strength using Django's validators."""
        validate_password(value)
        return value
    
    def validate(self, attrs):
        """
        Cross-field validation.
        Ensure required role-specific fields are provided.
        """
        role = attrs.get('role')
        
        # Specialist should have specialty
        if role == UserRole.SPECIALIST:
            if not attrs.get('specialty'):
                raise serializers.ValidationError({
                    'specialty': 'Specialty is required for Specialist role.'
                })
        
        # GP and Specialist should have license_number
        if role in [UserRole.GP, UserRole.SPECIALIST]:
            if not attrs.get('license_number'):
                raise serializers.ValidationError({
                    'license_number': f'License number is required for {role} role.'
                })
        
        return attrs
    
    def create(self, validated_data):
        """
        Create User and UserProfile atomically.
        """
        # Extract profile-specific fields
        specialty = validated_data.pop('specialty', None)
        department = validated_data.pop('department', None)
        license_number = validated_data.pop('license_number', None)
        
        # Create user (password will be hashed by UserManager)
        user = User.objects.create_user(
            email=validated_data['email'],
            password=validated_data['password'],
            first_name=validated_data['first_name'],
            last_name=validated_data['last_name'],
            role=validated_data['role'],
            organization=validated_data.get('organization', ''),
        )
        
        # Create profile with role-specific data
        UserProfile.objects.create(
            user=user,
            specialty=specialty or '',
            department=department or '',
            license_number=license_number or '',
        )
        
        return user


# =============================================================================
# LOGIN SERIALIZER
# =============================================================================

class LoginSerializer(serializers.Serializer):
    """
    Serializer for user login.
    
    Accepts:
    - email
    - password
    
    Returns:
    - User object (if credentials are valid)
    """
    
    email = serializers.EmailField()
    password = serializers.CharField(
        write_only=True,
        style={'input_type': 'password'}
    )
    
    def validate(self, attrs):
        """
        Validate credentials and return user.
        """
        email = attrs.get('email', '').lower()
        password = attrs.get('password', '')
        
        if not email or not password:
            raise serializers.ValidationError("Email and password are required.")
        
        # Authenticate user
        user = authenticate(
            request=self.context.get('request'),
            username=email,  # Django auth uses 'username' but we set USERNAME_FIELD to email
            password=password
        )
        
        if not user:
            raise serializers.ValidationError("Invalid email or password.")
        
        if not user.is_active:
            raise serializers.ValidationError("User account is disabled.")
        
        attrs['user'] = user
        return attrs

