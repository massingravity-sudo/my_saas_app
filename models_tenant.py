"""
models_tenant.py
────────────────
Nouveaux modèles multi-tenant à ajouter à CommSight.
Importe les mêmes `db` que models.py — aucun conflit.

Architecture :
  Tenant  →  1 entreprise  (isolée)
  TenantSubscription  →  abonnement actif du tenant
  TenantUser  →  lie un User à un Tenant (role tenant)
  SubscriptionPlan  →  définition des plans (mensuel/annuel/vie)
  TenantInvitation  →  lien d'invitation sécurisé
  TenantAuditLog  →  traçabilité RGPD par tenant
"""

from datetime import datetime
from models import db          # ← réutilise le même SQLAlchemy


# ═══════════════════════════════════════════════════════
# 1. TENANT  (une entreprise = un tenant)
# ═══════════════════════════════════════════════════════

class Tenant(db.Model):
    __tablename__ = 'tenants'

    id           = db.Column(db.Integer,     primary_key=True)
    name         = db.Column(db.String(120),  nullable=False)          # Nom entreprise
    slug         = db.Column(db.String(80),   unique=True, nullable=False)  # ex: acme-corp
    country      = db.Column(db.String(60),   default='')
    industry     = db.Column(db.String(80),   default='')
    logo_url     = db.Column(db.String(255),  default='')
    email_domain = db.Column(db.String(120),  default='')              # ex: acme.com
    status       = db.Column(db.String(20),   default='active')        # active | suspended | cancelled
    created_at   = db.Column(db.DateTime,     default=datetime.utcnow)
    updated_at   = db.Column(db.DateTime,     default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relations
    subscription = db.relationship('TenantSubscription', back_populates='tenant', uselist=False, cascade='all, delete-orphan')
    memberships  = db.relationship('TenantUser',         back_populates='tenant', cascade='all, delete-orphan')
    invitations  = db.relationship('TenantInvitation',   back_populates='tenant', cascade='all, delete-orphan')
    audit_logs   = db.relationship('TenantAuditLog',     back_populates='tenant', cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id':           self.id,
            'name':         self.name,
            'slug':         self.slug,
            'country':      self.country,
            'industry':     self.industry,
            'logo_url':     self.logo_url,
            'email_domain': self.email_domain,
            'status':       self.status,
            'created_at':   self.created_at.isoformat(),
            'subscription': self.subscription.to_dict() if self.subscription else None,
            'member_count': len(self.memberships),
        }


# ═══════════════════════════════════════════════════════
# 2. PLAN D'ABONNEMENT  (définition des plans)
# ═══════════════════════════════════════════════════════

class SubscriptionPlan(db.Model):
    __tablename__ = 'subscription_plans'

    id              = db.Column(db.Integer,    primary_key=True)
    code            = db.Column(db.String(30), unique=True, nullable=False)  # monthly | annual | lifetime
    name            = db.Column(db.String(60), nullable=False)
    price_eur       = db.Column(db.Float,      nullable=False)               # Prix en euros
    billing_period  = db.Column(db.String(20), default='monthly')            # monthly | yearly | once
    max_employees   = db.Column(db.Integer,    default=50)                   # -1 = illimité
    max_departments = db.Column(db.Integer,    default=5)                    # -1 = illimité
    max_storage_gb  = db.Column(db.Integer,    default=5)
    features        = db.Column(db.JSON,       default=list)                 # liste de feature keys
    is_active       = db.Column(db.Boolean,    default=True)
    created_at      = db.Column(db.DateTime,   default=datetime.utcnow)

    subscriptions   = db.relationship('TenantSubscription', back_populates='plan')

    def to_dict(self):
        return {
            'id':              self.id,
            'code':            self.code,
            'name':            self.name,
            'price_eur':       self.price_eur,
            'billing_period':  self.billing_period,
            'max_employees':   self.max_employees,
            'max_departments': self.max_departments,
            'max_storage_gb':  self.max_storage_gb,
            'features':        self.features or [],
        }


# ═══════════════════════════════════════════════════════
# 3. ABONNEMENT ACTIF D'UN TENANT
# ═══════════════════════════════════════════════════════

class TenantSubscription(db.Model):
    __tablename__ = 'tenant_subscriptions'

    id              = db.Column(db.Integer,   primary_key=True)
    tenant_id       = db.Column(db.Integer,   db.ForeignKey('tenants.id'), unique=True, nullable=False)
    plan_id         = db.Column(db.Integer,   db.ForeignKey('subscription_plans.id'), nullable=False)
    status          = db.Column(db.String(20), default='trialing')   # trialing | active | past_due | cancelled
    trial_ends_at   = db.Column(db.DateTime,  nullable=True)
    current_period_start = db.Column(db.DateTime, default=datetime.utcnow)
    current_period_end   = db.Column(db.DateTime, nullable=True)     # None = lifetime
    cancelled_at    = db.Column(db.DateTime,  nullable=True)
    payment_ref     = db.Column(db.String(120), default='')          # réf paiement externe
    created_at      = db.Column(db.DateTime,  default=datetime.utcnow)
    updated_at      = db.Column(db.DateTime,  default=datetime.utcnow, onupdate=datetime.utcnow)

    tenant = db.relationship('Tenant',           back_populates='subscription')
    plan   = db.relationship('SubscriptionPlan', back_populates='subscriptions')

    @property
    def is_active(self):
        if self.status == 'cancelled':
            return False
        if self.plan and self.plan.billing_period == 'once':
            return True            # lifetime : jamais expiré
        if self.status == 'trialing' and self.trial_ends_at:
            return datetime.utcnow() < self.trial_ends_at
        if self.current_period_end:
            return datetime.utcnow() < self.current_period_end
        return self.status == 'active'

    @property
    def days_remaining(self):
        if self.plan and self.plan.billing_period == 'once':
            return -1             # -1 = illimité
        end = self.trial_ends_at if self.status == 'trialing' else self.current_period_end
        if not end:
            return 0
        delta = (end - datetime.utcnow()).days
        return max(0, delta)

    def to_dict(self):
        return {
            'id':                    self.id,
            'plan':                  self.plan.to_dict() if self.plan else None,
            'status':                self.status,
            'is_active':             self.is_active,
            'days_remaining':        self.days_remaining,
            'trial_ends_at':         self.trial_ends_at.isoformat() if self.trial_ends_at else None,
            'current_period_start':  self.current_period_start.isoformat() if self.current_period_start else None,
            'current_period_end':    self.current_period_end.isoformat() if self.current_period_end else None,
            'cancelled_at':          self.cancelled_at.isoformat() if self.cancelled_at else None,
        }


# ═══════════════════════════════════════════════════════
# 4. LIEN USER ↔ TENANT  (remplace le champ company)
# ═══════════════════════════════════════════════════════

class TenantUser(db.Model):
    __tablename__ = 'tenant_users'

    id         = db.Column(db.Integer,   primary_key=True)
    tenant_id  = db.Column(db.Integer,   db.ForeignKey('tenants.id'), nullable=False)
    user_id    = db.Column(db.Integer,   db.ForeignKey('users.id'),   nullable=False)
    role       = db.Column(db.String(20), default='employee')          # tenant_admin | employee
    joined_at  = db.Column(db.DateTime,  default=datetime.utcnow)
    is_active  = db.Column(db.Boolean,   default=True)

    __table_args__ = (db.UniqueConstraint('tenant_id', 'user_id'),)

    tenant = db.relationship('Tenant', back_populates='memberships')
    # user   = db.relationship('User') ← ajouter si nécessaire

    def to_dict(self):
        return {
            'tenant_id': self.tenant_id,
            'user_id':   self.user_id,
            'role':      self.role,
            'joined_at': self.joined_at.isoformat(),
            'is_active': self.is_active,
        }


# ═══════════════════════════════════════════════════════
# 5. INVITATION PAR LIEN SÉCURISÉ
# ═══════════════════════════════════════════════════════

class TenantInvitation(db.Model):
    __tablename__ = 'tenant_invitations'

    id          = db.Column(db.Integer,    primary_key=True)
    tenant_id   = db.Column(db.Integer,    db.ForeignKey('tenants.id'), nullable=False)
    email       = db.Column(db.String(120), nullable=False)
    token       = db.Column(db.String(64),  unique=True, nullable=False)
    role        = db.Column(db.String(20),  default='employee')
    department  = db.Column(db.String(80),  default='')
    invited_by  = db.Column(db.Integer,    db.ForeignKey('users.id'), nullable=True)
    accepted    = db.Column(db.Boolean,    default=False)
    expires_at  = db.Column(db.DateTime,   nullable=False)
    created_at  = db.Column(db.DateTime,   default=datetime.utcnow)

    tenant = db.relationship('Tenant', back_populates='invitations')

    @property
    def is_valid(self):
        return not self.accepted and datetime.utcnow() < self.expires_at

    def to_dict(self):
        return {
            'id':         self.id,
            'tenant_id':  self.tenant_id,
            'email':      self.email,
            'token':      self.token,
            'role':       self.role,
            'department': self.department,
            'accepted':   self.accepted,
            'is_valid':   self.is_valid,
            'expires_at': self.expires_at.isoformat(),
            'created_at': self.created_at.isoformat(),
        }


# ═══════════════════════════════════════════════════════
# 6. AUDIT LOG PAR TENANT  (RGPD)
# ═══════════════════════════════════════════════════════

class TenantAuditLog(db.Model):
    __tablename__ = 'tenant_audit_logs'

    id         = db.Column(db.Integer,    primary_key=True)
    tenant_id  = db.Column(db.Integer,    db.ForeignKey('tenants.id'), nullable=False)
    user_id    = db.Column(db.Integer,    db.ForeignKey('users.id'),   nullable=True)
    action     = db.Column(db.String(80), nullable=False)
    details    = db.Column(db.Text,       default='')
    ip_address = db.Column(db.String(45), default='')
    timestamp  = db.Column(db.DateTime,   default=datetime.utcnow)

    tenant = db.relationship('Tenant', back_populates='audit_logs')

    def to_dict(self):
        return {
            'id':         self.id,
            'tenant_id':  self.tenant_id,
            'user_id':    self.user_id,
            'action':     self.action,
            'details':    self.details,
            'ip_address': self.ip_address,
            'timestamp':  self.timestamp.isoformat(),
        }