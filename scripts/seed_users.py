from sqlmodel import Session, select
from apps.api.core.db import engine
from apps.api.core.security import hash_password
from apps.api.models import User

DEMO_USERS = [
    ("admin@demo.com", "admin123", 2),
    ("public@demo.com", "public123", 0),
    ("internal@demo.com", "internal123", 1),
    ("restricted@demo.com", "restricted123", 2),
]

def main():
    with Session(engine) as session:
        for email, pw, level in DEMO_USERS:
            existing = session.exec(select(User).where(User.email == email)).first()
            if existing:
                print(f"User exists: {email}")
                continue
            u = User(
                email=email,
                hashed_password=hash_password(pw),
                max_access_level=level,
            )

            session.add(u)
            session.commit()
            print(f"Created: {email} (level={level})")

if __name__ == "__main__":
    main()
