"""
Custom middleware for Fish Recognition API
"""
from django.conf import settings
from django.utils.deprecation import MiddlewareMixin


class DisableCSRFForAPIMiddleware(MiddlewareMixin):
    """
    Disable CSRF protection for API endpoints
    """
    def process_request(self, request):
        if hasattr(settings, 'CSRF_EXEMPT_URLS'):
            for pattern in settings.CSRF_EXEMPT_URLS:
                if pattern.match(request.path_info.lstrip('/')):
                    setattr(request, '_dont_enforce_csrf_checks', True)
                    break
        return None
