"""
tenant_routes.py
────────────────
Blueprint Flask complet pour la gestion multi-tenant CommSight.

Routes exposées :
  POST   /api/tenants/register          Créer un espace entreprise (+ admin)
  POST   /api/tenants/login             Connexion tenant-aware
  GET    /api/tenants/me                Infos tenant de l'utilisateur connecté
  PUT    /api/tenants/me                Mettre à jour le profil entreprise
  GET    /api/tenants/me/subscription   Abonnement actif
  POST   /api/tenants/me/subscription/upgrade  Changer de plan
  DELETE /api/tenants/me/subscription   Résilier l'abonnement

  GET    /api/tenants/me/members        Liste membres
  POST   /api/tenants/invite            Inviter un employé
  POST   /api/tenants/invite/accept     Accepter une invitation
  DELETE /api/tenants/me/members/<id>   Retirer un membre

  GET    /api/plans                     Plans disponibles (public)

  # Super-admin (platform owner)
  GET    /api/superadmin/tenants        Tous les tenants
  GET    /api/superadmin/stats          Stats globales plateforme
  PUT    /api/superadmin/tenants/<id>/status  Suspendre/réactiver
"""

import secrets
from datetime import datetime, timedelta
from functools import wraps

from flask import Blueprint, jsonify, request, current_app
from models import db, User, Notification
from models_tenant import (
    Tenant, TenantSubscription, TenantUser,
    SubscriptionPlan, TenantInvitation, TenantAuditLog
)

tenant_bp = Blueprint('tenant', __name__)

# ─── Clé super-admin (à mettre en variable d'env en prod) ─────────────────
SUPER_ADMIN_KEY = 'COMMSIGHT_SUPER_KEY_2026'   # os.environ.get('SUPER_ADMIN_KEY')

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _get_token():
    return request.headers.get('Authorization', '').replace('Bearer ', '').strip()

def _user_from_token():
    """Retrouve l'utilisateur depuis le dict sessions de app.py."""
    token = _get_token()
    sessions = current_app.config.get('sessions', {})
    uid = sessions.get(token)
    if uid is None:
        return None
    return db.session.get(User, uid)

def _get_tenant_for_user(user):
    """Retourne le Tenant de cet utilisateur (via TenantUser)."""
    tm = TenantUser.query.filter_by(user_id=user.id, is_active=True).first()
    if not tm:
        return None
    return db.session.get(Tenant, tm.tenant_id)

def _get_tenant_role(user, tenant):
    tm = TenantUser.query.filter_by(user_id=user.id, tenant_id=tenant.id).first()
    return tm.role if tm else None

def _audit(tenant_id, user_id, action, details=''):
    log = TenantAuditLog(
        tenant_id  = tenant_id,
        user_id    = user_id,
        action     = action,
        details    = details,
        ip_address = request.remote_addr or '',
    )
    db.session.add(log)
    # Ne pas commit ici — le caller commit

def _slugify(text):
    import re
    s = re.sub(r'[^\w\s-]', '', text.lower())
    s = re.sub(r'[\s_-]+', '-', s).strip('-')
    return s

def _unique_slug(name):
    base = _slugify(name)[:60]
    slug = base
    i = 1
    while Tenant.query.filter_by(slug=slug).first():
        slug = f"{base}-{i}"
        i += 1
    return slug

def require_tenant_auth(f):
    """Vérifie auth + tenant actif."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = _user_from_token()
        if not user:
            return jsonify({'error': 'Non authentifié'}), 401
        tenant = _get_tenant_for_user(user)
        if not tenant:
            return jsonify({'error': 'Aucun espace entreprise associé'}), 403
        if tenant.status == 'suspended':
            return jsonify({'error': 'Espace entreprise suspendu — contactez le support'}), 403
        return f(user, tenant, *args, **kwargs)
    return wrapper

def require_tenant_admin(f):
    """Vérifie auth + rôle tenant_admin."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        user = _user_from_token()
        if not user:
            return jsonify({'error': 'Non authentifié'}), 401
        tenant = _get_tenant_for_user(user)
        if not tenant:
            return jsonify({'error': 'Aucun espace entreprise associé'}), 403
        if tenant.status == 'suspended':
            return jsonify({'error': 'Espace suspendu'}), 403
        role = _get_tenant_role(user, tenant)
        if role != 'tenant_admin':
            return jsonify({'error': 'Droits administrateur requis'}), 403
        return f(user, tenant, *args, **kwargs)
    return wrapper

def require_super_admin(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        key = request.headers.get('X-Super-Admin-Key', '')
        if key != SUPER_ADMIN_KEY:
            return jsonify({'error': 'Accès refusé'}), 403
        return f(*args, **kwargs)
    return wrapper

def _check_subscription_limit(tenant, resource):
    """
    Vérifie si le tenant peut encore créer une ressource.
    resource: 'employee' | 'department'
    Retourne (ok: bool, message: str)
    """
    sub = tenant.subscription
    if not sub or not sub.is_active:
        return False, "Abonnement inactif ou expiré"

    plan = sub.plan
    if not plan:
        return False, "Plan introuvable"

    if resource == 'employee':
        if plan.max_employees == -1:
            return True, ''
        current = TenantUser.query.filter_by(tenant_id=tenant.id, is_active=True).count()
        if current >= plan.max_employees:
            return False, f"Limite employés atteinte ({plan.max_employees}). Passez à un plan supérieur."

    elif resource == 'department':
        if plan.max_departments == -1:
            return True, ''
        # On compte les départements distincts des users du tenant
        from models import User as U
        user_ids = [tm.user_id for tm in TenantUser.query.filter_by(tenant_id=tenant.id).all()]
        depts = db.session.query(U.department).filter(U.id.in_(user_ids)).distinct().count()
        if depts >= plan.max_departments:
            return False, f"Limite départements atteinte ({plan.max_departments}). Passez à un plan supérieur."

    return True, ''

def _has_feature(tenant, feature_key):
    """Vérifie si le plan du tenant inclut une feature."""
    sub = tenant.subscription
    if not sub or not sub.is_active:
        return False
    plan = sub.plan
    if not plan:
        return False
    return feature_key in (plan.features or [])


# ═══════════════════════════════════════════════════════════════
# PLANS PUBLICS
# ═══════════════════════════════════════════════════════════════

@tenant_bp.route('/api/plans', methods=['GET'])
def get_plans():
    """Retourne tous les plans actifs — route publique (pas d'auth)."""
    plans = SubscriptionPlan.query.filter_by(is_active=True).all()
    return jsonify([p.to_dict() for p in plans])


# ═══════════════════════════════════════════════════════════════
# INSCRIPTION TENANT  (nouvelle entreprise)
# ═══════════════════════════════════════════════════════════════

@tenant_bp.route('/api/tenants/register', methods=['POST'])
def register_tenant():
    """
    Crée un espace entreprise + compte admin + abonnement essai 14j.
    Body JSON :
      company_name, admin_email, admin_full_name, admin_password,
      country (opt), industry (opt), plan_code (opt, défaut: monthly)
    """
    data = request.json or {}

    company_name    = data.get('company_name', '').strip()
    admin_email     = data.get('admin_email', '').strip()
    admin_full_name = data.get('admin_full_name', '').strip()
    admin_password  = data.get('admin_password', '')
    plan_code       = data.get('plan_code', 'monthly')

    # ── Validation basique ──
    if not company_name:
        return jsonify({'error': 'Nom de l\'entreprise requis'}), 400
    if not admin_email or '@' not in admin_email:
        return jsonify({'error': 'Email administrateur invalide'}), 400
    if not admin_full_name:
        return jsonify({'error': 'Nom complet requis'}), 400
    if len(admin_password) < 8:
        return jsonify({'error': 'Mot de passe : 8 caractères minimum'}), 400

    # ── Email déjà utilisé ? ──
    if User.query.filter_by(email=admin_email).first():
        return jsonify({'error': 'Cet email est déjà enregistré'}), 409

    # ── Plan ──
    plan = SubscriptionPlan.query.filter_by(code=plan_code, is_active=True).first()
    if not plan:
        plan = SubscriptionPlan.query.filter_by(code='monthly', is_active=True).first()
    if not plan:
        return jsonify({'error': 'Aucun plan disponible'}), 500

    try:
        # 1. Créer le tenant
        tenant = Tenant(
            name         = company_name,
            slug         = _unique_slug(company_name),
            country      = data.get('country', ''),
            industry     = data.get('industry', ''),
            email_domain = admin_email.split('@')[1] if '@' in admin_email else '',
            status       = 'active',
        )
        db.session.add(tenant)
        db.session.flush()   # on a besoin de tenant.id

        # 2. Créer l'admin
        admin = User(
            email          = admin_email,
            password       = admin_password,
            full_name      = admin_full_name,
            role           = 'admin',
            department     = 'Direction',
            position       = 'Administrateur',
            email_verified = True,
        )
        db.session.add(admin)
        db.session.flush()

        # 3. Lier admin → tenant
        membership = TenantUser(
            tenant_id = tenant.id,
            user_id   = admin.id,
            role      = 'tenant_admin',
        )
        db.session.add(membership)

        # 4. Abonnement essai 14 jours
        subscription = TenantSubscription(
            tenant_id       = tenant.id,
            plan_id         = plan.id,
            status          = 'trialing',
            trial_ends_at   = datetime.utcnow() + timedelta(days=14),
            current_period_start = datetime.utcnow(),
            current_period_end   = (
                None if plan.billing_period == 'once'
                else datetime.utcnow() + timedelta(days=14)
            ),
        )
        db.session.add(subscription)

        # 5. Notification de bienvenue
        notif = Notification(
            user_id = admin.id,
            title   = 'Bienvenue sur CommSight !',
            message = f'Votre espace "{company_name}" est prêt. Essai gratuit de 14 jours activé.',
            type    = 'success',
        )
        db.session.add(notif)

        # 6. Audit
        _audit(tenant.id, admin.id, 'tenant_created', f'Entreprise: {company_name}')

        db.session.commit()

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f'register_tenant error: {e}')
        return jsonify({'error': 'Erreur interne lors de la création'}), 500

    # ── Token de session ──
    token = f"token_{admin.id}_{datetime.utcnow().timestamp()}"
    if 'sessions' not in current_app.config:
        current_app.config['sessions'] = {}
    current_app.config['sessions'][token] = admin.id

    return jsonify({
        'message':    'Espace entreprise créé avec succès',
        'token':      token,
        'user':       admin.to_dict(),
        'tenant':     tenant.to_dict(),
        'trial_days': 14,
    }), 201


# ═══════════════════════════════════════════════════════════════
# INFOS TENANT COURANT
# ═══════════════════════════════════════════════════════════════

@tenant_bp.route('/api/tenants/me', methods=['GET'])
@require_tenant_auth
def get_my_tenant(user, tenant):
    role = _get_tenant_role(user, tenant)
    return jsonify({
        'tenant': tenant.to_dict(),
        'role':   role,
    })


@tenant_bp.route('/api/tenants/me', methods=['PUT'])
@require_tenant_admin
def update_my_tenant(user, tenant):
    data = request.json or {}
    for field in ('name', 'country', 'industry', 'logo_url', 'email_domain'):
        if field in data and data[field]:
            setattr(tenant, field, data[field].strip())

    tenant.updated_at = datetime.utcnow()
    _audit(tenant.id, user.id, 'tenant_updated', str(data))
    db.session.commit()
    return jsonify({'message': 'Profil entreprise mis à jour', 'tenant': tenant.to_dict()})


# ═══════════════════════════════════════════════════════════════
# ABONNEMENT
# ═══════════════════════════════════════════════════════════════

@tenant_bp.route('/api/tenants/me/subscription', methods=['GET'])
@require_tenant_auth
def get_my_subscription(user, tenant):
    sub = tenant.subscription
    if not sub:
        return jsonify({'error': 'Aucun abonnement'}), 404
    return jsonify(sub.to_dict())


@tenant_bp.route('/api/tenants/me/subscription/upgrade', methods=['POST'])
@require_tenant_admin
def upgrade_subscription(user, tenant):
    """
    Change le plan du tenant.
    Body: { plan_code: 'annual' | 'lifetime' | 'monthly', payment_ref: '...' }
    """
    data      = request.json or {}
    plan_code = data.get('plan_code')
    plan      = SubscriptionPlan.query.filter_by(code=plan_code, is_active=True).first()
    if not plan:
        return jsonify({'error': 'Plan introuvable'}), 404

    sub = tenant.subscription
    if not sub:
        sub = TenantSubscription(tenant_id=tenant.id)
        db.session.add(sub)

    sub.plan_id     = plan.id
    sub.status      = 'active'
    sub.payment_ref = data.get('payment_ref', '')
    sub.current_period_start = datetime.utcnow()

    if plan.billing_period == 'once':          # à vie
        sub.current_period_end = None
    elif plan.billing_period == 'yearly':
        sub.current_period_end = datetime.utcnow() + timedelta(days=365)
    else:                                        # mensuel
        sub.current_period_end = datetime.utcnow() + timedelta(days=30)

    sub.trial_ends_at = None
    sub.cancelled_at  = None
    sub.updated_at    = datetime.utcnow()

    _audit(tenant.id, user.id, 'subscription_upgraded', f'Nouveau plan: {plan_code}')

    # Notification
    notif = Notification(
        user_id = user.id,
        title   = 'Abonnement mis à jour',
        message = f'Votre plan "{plan.name}" est actif.',
        type    = 'success',
    )
    db.session.add(notif)
    db.session.commit()

    return jsonify({'message': 'Abonnement mis à jour', 'subscription': sub.to_dict()})


@tenant_bp.route('/api/tenants/me/subscription', methods=['DELETE'])
@require_tenant_admin
def cancel_subscription(user, tenant):
    sub = tenant.subscription
    if not sub:
        return jsonify({'error': 'Aucun abonnement actif'}), 404
    sub.status       = 'cancelled'
    sub.cancelled_at = datetime.utcnow()
    _audit(tenant.id, user.id, 'subscription_cancelled', '')
    db.session.commit()
    return jsonify({'message': 'Abonnement résilié. Accès conservé jusqu\'à la fin de la période.'})


# ═══════════════════════════════════════════════════════════════
# MEMBRES
# ═══════════════════════════════════════════════════════════════

@tenant_bp.route('/api/tenants/me/members', methods=['GET'])
@require_tenant_auth
def get_members(user, tenant):
    members = []
    for tm in TenantUser.query.filter_by(tenant_id=tenant.id, is_active=True).all():
        u = db.session.get(User, tm.user_id)
        if u:
            d = u.to_dict()
            d['tenant_role'] = tm.role
            d['joined_at']   = tm.joined_at.isoformat()
            members.append(d)
    return jsonify(members)


@tenant_bp.route('/api/tenants/me/members/<int:member_id>', methods=['DELETE'])
@require_tenant_admin
def remove_member(user, tenant, member_id):
    if member_id == user.id:
        return jsonify({'error': 'Vous ne pouvez pas vous retirer vous-même'}), 400
    tm = TenantUser.query.filter_by(tenant_id=tenant.id, user_id=member_id).first()
    if not tm:
        return jsonify({'error': 'Membre non trouvé'}), 404
    tm.is_active = False
    _audit(tenant.id, user.id, 'member_removed', f'user_id={member_id}')
    db.session.commit()
    return jsonify({'message': 'Membre retiré de l\'espace entreprise'})


# ═══════════════════════════════════════════════════════════════
# INVITATIONS
# ═══════════════════════════════════════════════════════════════

@tenant_bp.route('/api/tenants/invite', methods=['POST'])
@require_tenant_admin
def invite_member(user, tenant):
    """
    Envoie une invitation par email.
    Body: { email, role (opt), department (opt) }
    """
    data  = request.json or {}
    email = data.get('email', '').strip()

    if not email or '@' not in email:
        return jsonify({'error': 'Email invalide'}), 400

    # Vérifier limite employés
    ok, msg = _check_subscription_limit(tenant, 'employee')
    if not ok:
        return jsonify({'error': msg}), 403

    # Invitation déjà en cours ?
    existing = TenantInvitation.query.filter_by(
        tenant_id=tenant.id, email=email, accepted=False
    ).first()
    if existing and existing.is_valid:
        return jsonify({'error': 'Une invitation est déjà en attente pour cet email'}), 409

    token = secrets.token_urlsafe(32)
    inv   = TenantInvitation(
        tenant_id  = tenant.id,
        email      = email,
        token      = token,
        role       = data.get('role', 'employee'),
        department = data.get('department', ''),
        invited_by = user.id,
        expires_at = datetime.utcnow() + timedelta(days=7),
    )
    db.session.add(inv)
    _audit(tenant.id, user.id, 'invitation_sent', f'email={email}')
    db.session.commit()

    # Envoyer l'email d'invitation
    invite_url = f"http://localhost:5173/register?invitation={token}"
    body = (
        f"Bonjour,\n\n"
        f"{user.full_name} vous invite à rejoindre l'espace CommSight de \"{tenant.name}\".\n\n"
        f"Cliquez sur ce lien pour créer votre compte :\n{invite_url}\n\n"
        f"Ce lien expire dans 7 jours.\n\n"
        f"L'équipe CommSight"
    )
    try:
        from app import send_email
        send_email(email, f'Invitation CommSight — {tenant.name}', body)
    except Exception as e:
        current_app.logger.warning(f'Email invitation non envoyé: {e}')

    return jsonify({
        'message':    f'Invitation envoyée à {email}',
        'token':      token,
        'expires_at': inv.expires_at.isoformat(),
    }), 201


@tenant_bp.route('/api/tenants/invite/accept', methods=['POST'])
def accept_invitation():
    """
    Crée le compte de l'invité et le lie au tenant.
    Body: { token, full_name, password, department (opt) }
    """
    data     = request.json or {}
    token    = data.get('token', '').strip()
    inv      = TenantInvitation.query.filter_by(token=token).first()

    if not inv:
        return jsonify({'error': 'Invitation introuvable'}), 404
    if not inv.is_valid:
        return jsonify({'error': 'Invitation expirée ou déjà utilisée'}), 410

    tenant = db.session.get(Tenant, inv.tenant_id)
    if not tenant or tenant.status != 'active':
        return jsonify({'error': 'Espace entreprise inactif'}), 403

    full_name = data.get('full_name', '').strip()
    password  = data.get('password', '')

    if not full_name:
        return jsonify({'error': 'Nom complet requis'}), 400
    if len(password) < 8:
        return jsonify({'error': 'Mot de passe trop court'}), 400

    # Email déjà enregistré ?
    existing_user = User.query.filter_by(email=inv.email).first()
    if existing_user:
        # L'utilisateur existe déjà — on l'ajoute juste au tenant
        user = existing_user
    else:
        user = User(
            email          = inv.email,
            password       = password,
            full_name      = full_name,
            role           = 'employee',
            department     = data.get('department', inv.department) or 'Général',
            position       = data.get('position', 'Employé'),
            email_verified = True,
        )
        db.session.add(user)
        db.session.flush()

    # Lier au tenant
    already_linked = TenantUser.query.filter_by(tenant_id=tenant.id, user_id=user.id).first()
    if not already_linked:
        tm = TenantUser(
            tenant_id = tenant.id,
            user_id   = user.id,
            role      = inv.role,
        )
        db.session.add(tm)

    inv.accepted = True
    _audit(tenant.id, user.id, 'invitation_accepted', f'email={inv.email}')
    db.session.commit()

    # Token de session auto-login
    token_sess = f"token_{user.id}_{datetime.utcnow().timestamp()}"
    if 'sessions' not in current_app.config:
        current_app.config['sessions'] = {}
    current_app.config['sessions'][token_sess] = user.id

    return jsonify({
        'message': 'Compte créé et lié à l\'espace entreprise',
        'token':   token_sess,
        'user':    user.to_dict(),
        'tenant':  tenant.to_dict(),
    }), 201


@tenant_bp.route('/api/tenants/invite/<token>', methods=['GET'])
def get_invitation_info(token):
    """Retourne les infos publiques d'une invitation (pour pré-remplir le formulaire)."""
    inv = TenantInvitation.query.filter_by(token=token).first()
    if not inv:
        return jsonify({'error': 'Invitation introuvable'}), 404
    if not inv.is_valid:
        return jsonify({'error': 'Invitation expirée ou déjà utilisée'}), 410

    tenant = db.session.get(Tenant, inv.tenant_id)
    return jsonify({
        'email':       inv.email,
        'company':     tenant.name if tenant else '',
        'department':  inv.department,
        'role':        inv.role,
        'expires_at':  inv.expires_at.isoformat(),
    })


# ═══════════════════════════════════════════════════════════════
# VÉRIFICATION DES LIMITES  (utilisé par app.py)
# ═══════════════════════════════════════════════════════════════

@tenant_bp.route('/api/tenants/me/check-feature', methods=['GET'])
@require_tenant_auth
def check_feature(user, tenant):
    """GET /api/tenants/me/check-feature?key=ai_analytics"""
    key = request.args.get('key', '')
    return jsonify({'has_feature': _has_feature(tenant, key), 'feature': key})


@tenant_bp.route('/api/tenants/me/check-limit', methods=['GET'])
@require_tenant_auth
def check_limit(user, tenant):
    """GET /api/tenants/me/check-limit?resource=employee"""
    resource = request.args.get('resource', '')
    ok, msg  = _check_subscription_limit(tenant, resource)
    return jsonify({'allowed': ok, 'message': msg})


# ═══════════════════════════════════════════════════════════════
# AUDIT LOG
# ═══════════════════════════════════════════════════════════════

@tenant_bp.route('/api/tenants/me/audit', methods=['GET'])
@require_tenant_admin
def get_audit_log(user, tenant):
    limit = min(int(request.args.get('limit', 50)), 200)
    logs  = TenantAuditLog.query\
              .filter_by(tenant_id=tenant.id)\
              .order_by(TenantAuditLog.timestamp.desc())\
              .limit(limit).all()
    return jsonify([l.to_dict() for l in logs])


# ═══════════════════════════════════════════════════════════════
# SUPER-ADMIN  (propriétaire de la plateforme)
# ═══════════════════════════════════════════════════════════════

@tenant_bp.route('/api/superadmin/tenants', methods=['GET'])
@require_super_admin
def superadmin_list_tenants():
    tenants = Tenant.query.order_by(Tenant.created_at.desc()).all()
    result  = []
    for t in tenants:
        d = t.to_dict()
        d['member_count'] = TenantUser.query.filter_by(tenant_id=t.id, is_active=True).count()
        result.append(d)
    return jsonify(result)


@tenant_bp.route('/api/superadmin/stats', methods=['GET'])
@require_super_admin
def superadmin_stats():
    plans = SubscriptionPlan.query.filter_by(is_active=True).all()
    stats = {
        'total_tenants':  Tenant.query.count(),
        'active_tenants': Tenant.query.filter_by(status='active').count(),
        'total_users':    User.query.count(),
        'by_plan': {},
        'trialing':  TenantSubscription.query.filter_by(status='trialing').count(),
        'cancelled': TenantSubscription.query.filter_by(status='cancelled').count(),
        'mrr_eur':   0.0,
    }
    for plan in plans:
        count = TenantSubscription.query.filter_by(plan_id=plan.id, status='active').count()
        stats['by_plan'][plan.code] = count
        if plan.billing_period == 'monthly':
            stats['mrr_eur'] += count * plan.price_eur
        elif plan.billing_period == 'yearly':
            stats['mrr_eur'] += count * (plan.price_eur / 12)
    stats['mrr_eur'] = round(stats['mrr_eur'], 2)
    return jsonify(stats)


@tenant_bp.route('/api/superadmin/tenants/<int:tenant_id>/status', methods=['PUT'])
@require_super_admin
def superadmin_set_status(tenant_id):
    tenant = db.session.get(Tenant, tenant_id)
    if not tenant:
        return jsonify({'error': 'Tenant non trouvé'}), 404
    new_status = request.json.get('status')
    if new_status not in ('active', 'suspended', 'cancelled'):
        return jsonify({'error': 'Statut invalide'}), 400
    tenant.status = new_status
    db.session.commit()
    return jsonify({'message': f'Tenant {tenant.name} → {new_status}', 'tenant': tenant.to_dict()})


@tenant_bp.route('/api/superadmin/plans', methods=['POST'])
@require_super_admin
def superadmin_create_plan():
    """Crée ou met à jour un plan d'abonnement."""
    data = request.json or {}
    code = data.get('code', '').strip()
    if not code:
        return jsonify({'error': 'Code plan requis'}), 400

    plan = SubscriptionPlan.query.filter_by(code=code).first()
    if plan:
        # Mise à jour
        for field in ('name','price_eur','billing_period','max_employees','max_departments','max_storage_gb','features'):
            if field in data:
                setattr(plan, field, data[field])
    else:
        plan = SubscriptionPlan(
            code            = code,
            name            = data.get('name', code),
            price_eur       = data.get('price_eur', 0),
            billing_period  = data.get('billing_period', 'monthly'),
            max_employees   = data.get('max_employees', 50),
            max_departments = data.get('max_departments', 5),
            max_storage_gb  = data.get('max_storage_gb', 5),
            features        = data.get('features', []),
        )
        db.session.add(plan)

    db.session.commit()
    return jsonify({'message': 'Plan sauvegardé', 'plan': plan.to_dict()}), 201