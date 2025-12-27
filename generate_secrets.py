import secrets
import os
from pathlib import Path

def generate_secret(length=64):
    """Generate a cryptographically secure random secret"""
    return secrets.token_urlsafe(length)

def create_env_file():
    """Create .env file with secure secrets"""
    env_path = Path(__file__).parent / ".env"
    
    # Check if .env already exists
    if env_path.exists():
        response = input(".env file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted. Keeping existing .env file.")
            return
    
    # Generate secrets
    jwt_secret = generate_secret(64)
    flask_secret = generate_secret(64)
    
    # Create .env content
    env_content = f"""# EHR Assistant Environment Configuration
# Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# ============ SECURITY KEYS (REQUIRED) ============
JWT_SECRET={jwt_secret}
FLASK_SECRET_KEY={flask_secret}

# ============ DATABASE CONFIGURATION (REQUIRED) ============
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_mysql_password_here
DB_NAME=ehr_system
DB_PORT=3306

# ============ EMAIL CONFIGURATION (Optional) ============
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password_here
SMTP_FROM=noreply@yourdomain.com

# ============ APPLICATION CONFIGURATION ============
APP_URL=http://localhost:5000
FLASK_ENV=development
ALLOWED_ORIGINS=http://localhost:5000,http://127.0.0.1:5000

# ============ FILE UPLOAD LIMITS ============
UPLOAD_MAX_SIZE=10485760

# ============ GEMINI AI (Optional) ============
GEMINI_API_KEY=
GEMINI_MODEL=text-bison-001
GEMINI_AUTH_BEARER=
GEMINI_BASE=https://generativelanguage.googleapis.com/v1beta2
"""
    
    # Write to file
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print("✅ .env file created successfully!")
    print(f"📁 Location: {env_path}")
    print("\n⚠️  IMPORTANT NEXT STEPS:")
    print("1. Open .env file")
    print("2. Update DB_PASSWORD with your MySQL password")
    print("3. (Optional) Configure email settings for password reset")
    print("4. Keep .env file secure - never commit to git!")
    print("\n✨ Your security keys have been generated.")
    print("   JWT_SECRET: Generated ✓")
    print("   FLASK_SECRET_KEY: Generated ✓")

def main():
    print("🔐 EHR Assistant - Security Keys Generator")
    print("=" * 50)
    create_env_file()

if __name__ == "__main__":
    main()
