from functools import wraps
from django.http import JsonResponse
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import AuthenticationFailed
import jwt
from django.conf import settings  # Import Django settings
from django.contrib.auth import get_user_model
from .models import CustomUser

def token_required(view_func):
    def decorator(request, *args, **kwargs):
        # Extract the Authorization header from the request
        auth_header = request.META.get('HTTP_AUTHORIZATION', '')

        # Ensure that the Authorization header is present
        if not auth_header:
            return JsonResponse({'error': 'Authorization header is missing'}, status=401)

        # Verify the Authorization header format
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            return JsonResponse({'error': 'Invalid Authorization header format'}, status=401)

        # Extract the token from the Authorization header
        token = parts[1]

        # Attempt to decode the token and return its payload
        try:
            payload = jwt.decode(token, key=settings.SECRET_KEY, algorithms=['HS256'])  # Use SECRET_KEY from settings
        except Exception as decode_error:
            return JsonResponse({'error': f'Failed to decode token: {decode_error}'}, status=401)

        # Extract user ID from token payload
        user_id_from_token = payload.get('user_id')

        # Compare user ID from token payload with user ID retrieved from the database
        user = CustomUser.objects.filter(id=user_id_from_token).first()

        # If user not found, return unauthorized response
        if not user:
            return JsonResponse({'error': 'User not found'}, status=401)

        # Pass the user object to the view function
        return view_func(request, user, *args, **kwargs)

    # Return the decorator
    return decorator

# def get_user_data_from_payload(token):
#     try:
#         payload = jwt.decode(token, key=settings.SECRET_KEY, algorithms=['HS256'])
#         print('Decoded token payloadddddd:', payload)
        
#         user_id = payload.get('user_id')
#         print('User ID from token payloaddddd:', user_id)
        
#         if user_id:
#             UserModel = get_user_model()
#             try:
#                 user = UserModel.objects.get(pk=user_id)
#                 return {
#                     'id': user.id,
#                     'email': user.email,
#                     'name': user.name,
#                     # Add other user attributes as needed
#                 }
#             except UserModel.DoesNotExist:
#                 return {'error': 'User with the provided ID does not exist'}
#         else:
#             return {'error': 'User ID not found in token payload'}
#     except Exception as e:
#         return {'error': f'Failed to decode token: {e}'}