"""
Local Offline Authentication Module
Simple JSON-based user management
"""

import json
import os
from datetime import datetime

class LocalAuth:
    """Local authentication without Firebase"""
    
    def __init__(self, users_file="users.json"):
        """Initialize with local users file"""
        self.users_file = users_file
        self.users = self._load_users()
        print("✅ Local authentication initialized")
    
    def _load_users(self):
        """Load users from JSON file"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading users: {e}")
                return {}
        return {}
    
    def authenticate(self, username, password):
        """
        Authenticate user locally
        
        Args:
            username: Username
            password: Password
        
        Returns:
            dict: User data if authenticated, None otherwise
        """
        if username not in self.users:
            return None
        
        user = self.users[username]
        if user.get('password') == password:
            return {
                'user_id': username,
                'username': username,
                'email': user.get('email', ''),
                'plant_name': user.get('plant_name', 'My Plant')
            }
        
        return None
    
    def get_user(self, username):
        """Get user data by username"""
        if username in self.users:
            return {
                'user_id': username,
                'username': username,
                'email': self.users[username].get('email', ''),
                'plant_name': self.users[username].get('plant_name', 'My Plant')
            }
        return None

# Global instance
_auth = None

def get_auth():
    """Get or create local auth instance"""
    global _auth
    if _auth is None:
        _auth = LocalAuth()
    return _auth
