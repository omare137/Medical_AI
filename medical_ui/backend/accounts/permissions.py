"""
Role-based Permission Classes for Medical Auth MVP.

These permission classes enforce role-based access control:
- IsGP: Only allows GP users
- IsSpecialist: Only allows Specialist users
- IsReceptionist: Only allows Receptionist users
- IsAdmin: Only allows Admin users

Usage:
    from accounts.permissions import IsGP
    
    class GPDashboardView(APIView):
        permission_classes = [IsAuthenticated, IsGP]
        ...
"""

from rest_framework.permissions import BasePermission
from .models import UserRole


class IsGP(BasePermission):
    """
    Permission class that only allows GP users.
    
    Used for endpoints that should only be accessible by General Practitioners.
    """
    message = "Access denied. This endpoint is only available to GPs."
    
    def has_permission(self, request, view):
        # User must be authenticated and have GP role
        return (
            request.user and
            request.user.is_authenticated and
            request.user.role == UserRole.GP
        )


class IsSpecialist(BasePermission):
    """
    Permission class that only allows Specialist users.
    
    Used for endpoints that should only be accessible by Specialists.
    """
    message = "Access denied. This endpoint is only available to Specialists."
    
    def has_permission(self, request, view):
        # User must be authenticated and have SPECIALIST role
        return (
            request.user and
            request.user.is_authenticated and
            request.user.role == UserRole.SPECIALIST
        )


class IsReceptionist(BasePermission):
    """
    Permission class that only allows Receptionist users.
    
    Used for endpoints that should only be accessible by Receptionists.
    """
    message = "Access denied. This endpoint is only available to Receptionists."
    
    def has_permission(self, request, view):
        # User must be authenticated and have RECEPTIONIST role
        return (
            request.user and
            request.user.is_authenticated and
            request.user.role == UserRole.RECEPTIONIST
        )


class IsAdmin(BasePermission):
    """
    Permission class that only allows Admin users.
    
    Used for administrative endpoints (future use).
    """
    message = "Access denied. This endpoint is only available to Administrators."
    
    def has_permission(self, request, view):
        # User must be authenticated and have ADMIN role
        return (
            request.user and
            request.user.is_authenticated and
            request.user.role == UserRole.ADMIN
        )


class IsGPOrSpecialist(BasePermission):
    """
    Permission class that allows either GP or Specialist users.
    
    Used for endpoints that should be accessible by medical professionals.
    """
    message = "Access denied. This endpoint is only available to GPs and Specialists."
    
    def has_permission(self, request, view):
        return (
            request.user and
            request.user.is_authenticated and
            request.user.role in [UserRole.GP, UserRole.SPECIALIST]
        )

