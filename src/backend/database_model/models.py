import uuid
from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def gen_uuid():
    return str(uuid.uuid4())


class User(db.Model):
    """Primary holder account."""
    __tablename__ = "users"

    id = db.Column(db.String(36), primary_key=True, default=gen_uuid)
    full_name = db.Column(db.String(200), nullable=False)
    country = db.Column(db.String(100), nullable=False)
    national_id = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    consent_status = db.Column(db.Boolean, default=False)
    consent_email_scan = db.Column(db.Boolean, default=False)
    gmail_oauth_token = db.Column(db.Text, nullable=True)  # we will encrypt this in prod

    is_deceased = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # deleting a primary holder cleans up everything related to it
    next_of_kin = db.relationship("NextOfKin", backref="primary_holder", lazy=True,
                                   cascade="all, delete-orphan")
    assets = db.relationship("Asset", backref="owner", lazy=True,
                              cascade="all, delete-orphan")
    fail_safe = db.relationship("FailSafe", backref="primary_holder", uselist=False,
                                 cascade="all, delete-orphan")
    death_record = db.relationship("DeathRecord", backref="user", uselist=False,
                                    cascade="all, delete-orphan")


class NextOfKin(db.Model):
    __tablename__ = "next_of_kin"

    id = db.Column(db.String(36), primary_key=True, default=gen_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)
    name = db.Column(db.String(200), nullable=False)
    contact = db.Column(db.String(200), nullable=False)  # phone or email
    national_id = db.Column(db.String(50), nullable=True)
    role = db.Column(db.String(20), default="Viewer")  # Admin | Viewer
    decision_status = db.Column(db.String(20), default="pending")  # pending | accepted | declined
    invite_token = db.Column(db.String(64), default=gen_uuid, unique=True)
    biometric_enrolled = db.Column(db.Boolean, default=False)
    biometric_verified_at = db.Column(db.DateTime(timezone=True), nullable=True)

    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    consent_form = db.relationship("ConsentForm", backref="next_of_kin", uselist=False,
                                    cascade="all, delete-orphan")
    notifications = db.relationship("NotificationLog", backref="next_of_kin", lazy=True,
                                     cascade="all, delete-orphan")


class FailSafe(db.Model):
    __tablename__ = "fail_safe"

    id = db.Column(db.String(36), primary_key=True, default=gen_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, unique=True)
    option = db.Column(db.String(20), default="charity")  # charity | government
    details = db.Column(db.Text, nullable=True)
    engaged = db.Column(db.Boolean, default=False)
    engaged_at = db.Column(db.DateTime(timezone=True), nullable=True)


class Asset(db.Model):
    __tablename__ = "assets"

    id = db.Column(db.String(36), primary_key=True, default=gen_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)
    asset_type = db.Column(db.String(100), nullable=False)  # digital assetsetc.
    label = db.Column(db.String(200), nullable=False)
    location = db.Column(db.String(200), nullable=True)
    value_estimate = db.Column(db.String(50), nullable=True) 
    source = db.Column(db.String(20), default="manual")  # manual, email_suggested, web_discovered
    status = db.Column(db.String(20), default="confirmed")  # confirmed, suggested, rejected
    date_created = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class DeathRecord(db.Model):
    """I will be using webhook to simulate the death registry."""
    __tablename__ = "death_records"

    id = db.Column(db.String(36), primary_key=True, default=gen_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, unique=True)
    cert_generated = db.Column(db.Boolean, default=False)
    cert_number = db.Column(db.String(50), nullable=True)
    cert_date = db.Column(db.DateTime, nullable=True)
    cert_file_url = db.Column(db.String(300), nullable=True)
    issuing_authority = db.Column(db.String(200), default="SIMULATED NATIONAL DEATH REGISTRY")
    received_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ConsentForm(db.Model):
    __tablename__ = "consent_forms"

    id = db.Column(db.String(36), primary_key=True, default=gen_uuid)
    next_of_kin_id = db.Column(db.String(36), db.ForeignKey("next_of_kin.id"), nullable=False, unique=True)
    downloaded = db.Column(db.Boolean, default=False)
    downloaded_at = db.Column(db.DateTime(timezone=True), nullable=True)
    legal_status = db.Column(db.String(20), default="pending")  # pending | engaged


class AuditLog(db.Model):
    __tablename__ = "audit_log"

    id = db.Column(db.String(36), primary_key=True, default=gen_uuid)
    user_id = db.Column(db.String(36), nullable=True)
    actor = db.Column(db.String(200), nullable=False)  # e.g. "System", "Admin"
    action = db.Column(db.String(300), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class NotificationLog(db.Model):
    __tablename__ = "notification_log"

    id = db.Column(db.String(36), primary_key=True, default=gen_uuid)
    next_of_kin_id = db.Column(db.String(36), db.ForeignKey("next_of_kin.id"), nullable=False)
    channel = db.Column(db.String(20), default="email")  # sms | email
    status = db.Column(db.String(20), default="sent")
    sent_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))