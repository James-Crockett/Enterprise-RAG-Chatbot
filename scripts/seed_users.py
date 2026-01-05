from sqlmodel import Session, select
from apps.api.core.db import engine
from apps.api.core.security import hash_password
from apps.api.models import User

DEMO_USERS = [
    # email, password, max_access_level, is_admin
    ("admin@demo.com", "admin123", 2, True),
    ("public@demo.com", "public123", 0, False),
    ("internal@demo.com", "internal123", 1, False),
    ("restricted@demo.com", "restricted123", 2, False),
]

def main():
    with Session(engine) as session:
        for email, pw, level, is_admin in DEMO_USERS:
            existing = session.exec(select(User).where(User.email == email)).first()
            if existing:
                print(f"User exists: {email}")
                continue
            u = User(
                email=email,
                hashed_password=hash_password(pw),
                max_access_level=level,
                is_admin=is_admin,
            )

            session.add(u)
            session.commit()
            print(f"Created: {email} (level={level}, admin={is_admin})")

if __name__ == "__main__":
    main()
