"""
Custom User Model and UserProfile for Medical Auth MVP.

Key Features:
- Custom User model with email as username
- Role-based access (GP, SPECIALIST, RECEPTIONIST, ADMIN)
- UserProfile for role-specific metadata
- Roles are immutable after creation
"""

from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.utils import timezone


# =============================================================================
# ROLE CHOICES (Enum-like)
# =============================================================================

class UserRole(models.TextChoices):
    """
    Hard-coded user roles for the medical platform.
    Stored in database, cannot be changed after signup.
    """
    GP = 'GP', 'General Practitioner'
    SPECIALIST = 'SPECIALIST', 'Specialist'
    RECEPTIONIST = 'RECEPTIONIST', 'Receptionist'
    ADMIN = 'ADMIN', 'Administrator'  # For future use


# =============================================================================
# CUSTOM USER MANAGER
# =============================================================================

class UserManager(BaseUserManager):
    """
    Custom manager for User model with email as the unique identifier.
    """
    
    def create_user(self, email, password=None, **extra_fields):
        """
        Create and save a regular user with the given email and password.
        """
        if not email:
            raise ValueError('Email address is required')
        
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)  # Hashes password using Django's default hasher
        user.save(using=self._db)
        return user
    
    def create_superuser(self, email, password=None, **extra_fields):
        """
        Create and save a superuser with the given email and password.
        """
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)
        extra_fields.setdefault('role', UserRole.ADMIN)
        
        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')
        
        return self.create_user(email, password, **extra_fields)


# =============================================================================
# CUSTOM USER MODEL
# =============================================================================

class User(AbstractBaseUser, PermissionsMixin):
    """
    Custom User model using email as the username field.
    
    Fields:
    - id: Auto-generated primary key
    - first_name: User's first name
    - last_name: User's last name
    - email: Unique email address (used for login)
    - password: Hashed password (inherited from AbstractBaseUser)
    - role: User's role in the system (immutable after creation)
    - organization: Optional organization name
    - is_active: Whether the user account is active
    - is_staff: Whether the user can access admin site
    - created_at: Account creation timestamp
    """
    
    # Basic fields
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.EmailField(unique=True, db_index=True)
    
    # Role field - immutable after creation
    # This is enforced at the serializer/view level
    role = models.CharField(
        max_length=20,
        choices=UserRole.choices,
        default=UserRole.GP
    )
    
    # Optional organization
    organization = models.CharField(max_length=255, blank=True, null=True)
    
    # Account status
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    
    # Timestamps
    created_at = models.DateTimeField(default=timezone.now)
    
    # Use email as the username field
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'last_name']
    
    # Custom manager
    objects = UserManager()
    
    class Meta:
        db_table = 'users'
        verbose_name = 'User'
        verbose_name_plural = 'Users'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.first_name} {self.last_name} ({self.email})"
    
    def get_full_name(self):
        """Return the user's full name."""
        return f"{self.first_name} {self.last_name}".strip()
    
    def get_short_name(self):
        """Return the user's first name."""
        return self.first_name
    
    # Role check helpers
    def is_gp(self):
        """Check if user is a GP."""
        return self.role == UserRole.GP
    
    def is_specialist(self):
        """Check if user is a Specialist."""
        return self.role == UserRole.SPECIALIST
    
    def is_receptionist(self):
        """Check if user is a Receptionist."""
        return self.role == UserRole.RECEPTIONIST
    
    def is_admin(self):
        """Check if user is an Admin."""
        return self.role == UserRole.ADMIN


# =============================================================================
# USER PROFILE MODEL (Role-specific metadata)
# =============================================================================

class UserProfile(models.Model):
    """
    Stores role-specific metadata for users.
    
    Only relevant fields are populated based on user's role:
    - GP: license_number
    - SPECIALIST: specialty, license_number
    - RECEPTIONIST: department
    - ADMIN: (no specific fields needed)
    """
    
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='profile'
    )
    
    # Specialist-specific
    specialty = models.CharField(max_length=100, blank=True, null=True)
    
    # Receptionist-specific
    department = models.CharField(max_length=100, blank=True, null=True)
    
    # GP and Specialist
    license_number = models.CharField(max_length=50, blank=True, null=True)
    
    class Meta:
        db_table = 'user_profiles'
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'
    
    def __str__(self):
        return f"Profile for {self.user.email}"

