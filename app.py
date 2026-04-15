from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import os
import json
import mimetypes
import hashlib
from flask_mail import Mail, Message as MailMessage
import secrets
import re
import logging
from models_tenant import (
    Tenant, TenantSubscription, TenantUser,
    SubscriptionPlan, TenantInvitation, TenantAuditLog
)

# ============================================
# CONFIGURATION
# ============================================

app = Flask(__name__)

# ── CORS ─────────────────────────────────────────────────────
CORS(app, origins=[
    "https://my-front-app-rust.vercel.app",
    "http://localhost:5173",
    "http://localhost:3000",
], supports_credentials=True)

# ── Base de données ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL',
    f"sqlite:///{os.path.join(BASE_DIR, 'commsight.db')}"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
}

# ── Upload ───────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'doc', 'docx', 'xls', 'xlsx'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ── Email ────────────────────────────────────────────────────
app.config['MAIL_SERVER']         = 'smtp.gmail.com'
app.config['MAIL_PORT']           = 587
app.config['MAIL_USE_TLS']        = True
app.config['MAIL_USE_SSL']        = False
app.config['MAIL_USERNAME']       = os.environ.get('MAIL_USERNAME', 'benslimanefaiz7@gmail.com')
app.config['MAIL_PASSWORD']       = os.environ.get('MAIL_PASSWORD', 'yaeylorbnbjsztsj')
app.config['MAIL_DEFAULT_SENDER'] = 'CommSight <benslimanefaiz7@gmail.com>'

mail = Mail(app)

# ── Initialisation DB ────────────────────────────────────────
from models import (
    db, User, LoginHistory, Post, Task, Leave,
    Notification, Activity, Conversation, Message,
    Survey, SurveyResponse, Feedback,
    DocumentFolder, Document
)
db.init_app(app)

# ── Modèles optionnels (chef de département) ─────────────────
try:
    from models import Evaluation, Prime, PosteOuvert, Candidat
    CHEF_MODELS_AVAILABLE = True
except ImportError:
    CHEF_MODELS_AVAILABLE = False
    print("⚠️  Modèles Evaluation/Prime/PosteOuvert/Candidat non disponibles")

# ── OTP storage (en mémoire, court-lived) ────────────────────
verification_codes: dict = {}

# ============================================
# UTILITAIRES
# ============================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_password_strength(password):
    if len(password) < 8:
        return False, "Le mot de passe doit contenir au moins 8 caractères"
    if not re.search(r'[A-Z]', password):
        return False, "Le mot de passe doit contenir au moins une majuscule"
    if not re.search(r'[a-z]', password):
        return False, "Le mot de passe doit contenir au moins une minuscule"
    if not re.search(r'\d', password):
        return False, "Le mot de passe doit contenir au moins un chiffre"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Le mot de passe doit contenir au moins un caractère spécial"
    return True, "Mot de passe valide"

def generate_otp():
    return ''.join([str(secrets.randbelow(10)) for _ in range(6)])

def send_email(to, subject, body):
    print(f"\n{'='*60}\n📧 Email → {to} | {subject}")
    try:
        msg = MailMessage(subject, recipients=[to])
        msg.body = body
        mail.send(msg)
        print("✅ Envoyé\n")
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}\n")
        return False

# ── Sessions (token → user_id) ───────────────────────────────
# Token stable basé sur user_id + password hash
# Survit aux redémarrages Railway — plus de déconnexions intempestives
sessions: dict[str, int] = {}

TOKEN_SECRET = os.environ.get('TOKEN_SECRET', 'commsight_secret_key_2025')

def generate_stable_token(user_id: int, password: str) -> str:
    """Génère un token déterministe qui survit aux redémarrages du serveur."""
    raw = f"{user_id}:{password}:{TOKEN_SECRET}"
    return hashlib.sha256(raw.encode()).hexdigest()

def get_user_from_token(token: str):
    # 1. Cherche en cache mémoire (rapide)
    uid = sessions.get(token)
    if uid:
        return db.session.get(User, uid)

    # 2. Après redémarrage : revalide le token contre la DB
    try:
        for user in User.query.all():
            if generate_stable_token(user.id, user.password) == token:
                sessions[token] = user.id  # remet en cache
                return user
    except Exception:
        pass
    return None

def require_auth(f):
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        user = get_user_from_token(token)
        if not user:
            return jsonify({'error': 'Non authentifié'}), 401
        return f(user, *args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

def require_admin(f):
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        user = get_user_from_token(token)
        if not user or user.role != 'admin':
            return jsonify({'error': 'Accès refusé - Droits administrateur requis'}), 403
        return f(user, *args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

def log_activity(user_id: int, action: str, details: str):
    act = Activity(user_id=user_id, action=action, details=details)
    db.session.add(act)
    db.session.commit()

def create_notification(user_id: int, title: str, message: str, type: str = 'info'):
    notif = Notification(user_id=user_id, title=title, message=message, type=type)
    db.session.add(notif)
    db.session.commit()
    return notif

def generate_task_code():
    year = datetime.now().year
    count = Task.query.filter(Task.code.like(f'TSK-{year}-%')).count()
    return f'TSK-{year}-{str(count + 1).zfill(4)}'

def _refresh_ml_config():
    """Synchronise les données live pour le moteur ML."""
    try:
        app.config['users']         = [u.to_dict(include_password=True) for u in User.query.all()]
        app.config['tasks']         = [t.to_dict() for t in Task.query.all()]
        app.config['messages']      = [m.to_dict() for m in Message.query.all()]
        app.config['leaves']        = [l.to_dict() for l in Leave.query.all()]
        app.config['activities']    = [a.to_dict() for a in Activity.query.order_by(Activity.timestamp.desc()).limit(500).all()]
        app.config['feedbacks']     = [f.to_dict() for f in Feedback.query.all()]
        app.config['conversations'] = [c.to_dict() for c in Conversation.query.all()]
    except Exception as e:
        print(f"⚠️  _refresh_ml_config erreur: {e}")

# ============================================
# MOTEUR ML — attrape TOUS les types d'erreurs
# ============================================

try:
    from ml_analytics_routes import ml_bp
    from mlops_layer import mlops_bp
    app.register_blueprint(ml_bp)
    app.register_blueprint(mlops_bp)
    ML_AVAILABLE = True
    print("✅ Moteur ML chargé")
except Exception as e:
    ML_AVAILABLE = False
    print(f"⚠️  ML non disponible: {e}")

# ============================================
# MULTI-TENANT
# ============================================

try:
    from tenant_routes import tenant_bp
    app.register_blueprint(tenant_bp)
    print("✅ Multi-tenant chargé")
except Exception as e:
    print(f"⚠️  Multi-tenant non disponible: {e}")

# ============================================
# AUTHENTIFICATION
# ============================================

@app.route('/api/login', methods=['POST'])
def login():
    data     = request.json
    email    = data.get('email', '').strip()
    password = data.get('password', '')

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'error': 'Email ou mot de passe incorrect'}), 401
    if user.login_attempts >= 5:
        return jsonify({'error': 'Compte bloqué. Contactez un administrateur'}), 403
    if user.password != password:
        user.login_attempts += 1
        db.session.commit()
        return jsonify({'error': 'Email ou mot de passe incorrect'}), 401

    user.login_attempts = 0
    user.last_login     = datetime.utcnow()
    db.session.commit()

    # Token stable — survit aux redémarrages
    token = generate_stable_token(user.id, user.password)
    sessions[token] = user.id
    app.config.setdefault('sessions', {})
    app.config['sessions'][token] = user.id

    history = LoginHistory(
        user_id    = user.id,
        email      = email,
        ip_address = request.remote_addr,
        user_agent = request.headers.get('User-Agent', 'Unknown')
    )
    db.session.add(history)
    db.session.commit()

    log_activity(user.id, 'login', f"Connexion de {user.full_name}")
    _refresh_ml_config()
    print(f"✅ Connexion: {user.full_name} ({user.role})")

    return jsonify({'token': token, 'user': user.to_dict()})

@app.route('/api/auth/register-request', methods=['POST'])
def register_request():
    data      = request.json
    email     = data.get('email', '').strip()
    full_name = data.get('full_name', '')

    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Cet email est déjà utilisé'}), 400
    if not email or '@' not in email:
        return jsonify({'error': 'Email invalide'}), 400

    code = generate_otp()
    verification_codes[email] = {
        'code': code,
        'expires': datetime.utcnow() + timedelta(minutes=10),
        'type': 'registration',
        'data': data
    }
    body = f"Bonjour {full_name},\n\nVotre code CommSight : {code}\n\nExpire dans 10 minutes.\n\nL'équipe CommSight"
    if send_email(email, 'CommSight - Code de vérification', body):
        return jsonify({'message': 'Code envoyé', 'email': email}), 200
    return jsonify({'error': "Erreur envoi email"}), 500

@app.route('/api/auth/verify-code', methods=['POST'])
def verify_code():
    data     = request.json
    email    = data.get('email', '')
    code     = data.get('code', '')
    password = data.get('password', '')

    if email not in verification_codes:
        return jsonify({'error': 'Aucun code en attente'}), 400
    stored = verification_codes[email]
    if datetime.utcnow() > stored['expires']:
        del verification_codes[email]
        return jsonify({'error': 'Code expiré'}), 400
    if stored['code'] != code:
        return jsonify({'error': 'Code incorrect'}), 400

    is_valid, msg = validate_password_strength(password)
    if not is_valid:
        return jsonify({'error': msg}), 400

    reg = stored['data']
    user = User(
        email          = email,
        password       = password,
        full_name      = reg.get('full_name'),
        role           = 'employee',
        department     = reg.get('department'),
        position       = reg.get('position', 'Employé'),
        phone          = reg.get('phone', ''),
        email_verified = True,
    )
    db.session.add(user)
    db.session.commit()
    del verification_codes[email]

    log_activity(user.id, 'account_created', f"Compte créé pour {user.full_name}")
    _refresh_ml_config()

    body = f"Bonjour {user.full_name},\n\nVotre compte CommSight est créé !\n\nEmail: {email}\nDépartement: {user.department}\n\nL'équipe CommSight"
    send_email(email, 'Bienvenue sur CommSight', body)

    return jsonify({'message': 'Compte créé', 'user': user.to_dict()}), 201

@app.route('/api/auth/forgot-password', methods=['POST'])
def forgot_password():
    email = request.json.get('email', '').strip()
    user  = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'message': 'Si cet email existe, un code a été envoyé'}), 200

    code = generate_otp()
    verification_codes[email] = {
        'code': code,
        'expires': datetime.utcnow() + timedelta(minutes=15),
        'type': 'password_reset',
        'user_id': user.id
    }
    body = f"Bonjour {user.full_name},\n\nCode de réinitialisation : {code}\n\nExpire dans 15 minutes.\n\nL'équipe CommSight"
    send_email(email, 'CommSight - Réinitialisation mot de passe', body)
    return jsonify({'message': 'Si cet email existe, un code a été envoyé'}), 200

@app.route('/api/auth/reset-password', methods=['POST'])
def reset_password():
    data         = request.json
    email        = data.get('email', '')
    code         = data.get('code', '')
    new_password = data.get('new_password', '')

    if email not in verification_codes:
        return jsonify({'error': 'Aucun code en attente'}), 400
    stored = verification_codes[email]
    if stored['type'] != 'password_reset':
        return jsonify({'error': 'Code invalide'}), 400
    if datetime.utcnow() > stored['expires']:
        del verification_codes[email]
        return jsonify({'error': 'Code expiré'}), 400
    if stored['code'] != code:
        return jsonify({'error': 'Code incorrect'}), 400

    is_valid, msg = validate_password_strength(new_password)
    if not is_valid:
        return jsonify({'error': msg}), 400

    user = db.session.get(User, stored['user_id'])
    if not user:
        return jsonify({'error': 'Utilisateur introuvable'}), 404

    user.password = new_password
    db.session.commit()
    del verification_codes[email]
    log_activity(user.id, 'password_reset', 'Mot de passe réinitialisé')

    body = f"Bonjour {user.full_name},\n\nVotre mot de passe a été modifié.\n\nSi ce n'est pas vous, contactez l'admin immédiatement.\n\nL'équipe CommSight"
    send_email(email, 'CommSight - Mot de passe modifié', body)
    return jsonify({'message': 'Mot de passe réinitialisé'}), 200

@app.route('/api/auth/resend-code', methods=['POST'])
def resend_code():
    email = request.json.get('email', '')
    if email not in verification_codes:
        return jsonify({'error': 'Aucune demande en cours'}), 400

    stored = verification_codes[email]
    code   = generate_otp()
    stored['code']    = code
    stored['expires'] = datetime.utcnow() + timedelta(minutes=10)

    if stored['type'] == 'registration':
        full_name = stored['data'].get('full_name', 'Utilisateur')
    else:
        u = db.session.get(User, stored.get('user_id'))
        full_name = u.full_name if u else 'Utilisateur'

    body = f"Bonjour {full_name},\n\nNouveau code CommSight : {code}\n\nExpire dans 10 minutes.\n\nL'équipe CommSight"
    send_email(email, 'CommSight - Nouveau code', body)
    return jsonify({'message': 'Nouveau code envoyé'}), 200

@app.route('/api/auth/login-history', methods=['GET'])
@require_auth
def get_login_history(user):
    history = LoginHistory.query.filter_by(user_id=user.id)\
                .order_by(LoginHistory.timestamp.desc()).limit(20).all()
    return jsonify([h.to_dict() for h in history])

@app.route('/api/auth/change-password', methods=['POST'])
@require_auth
def change_password(user):
    data             = request.json
    current_password = data.get('current_password', '')
    new_password     = data.get('new_password', '')

    if user.password != current_password:
        return jsonify({'error': 'Mot de passe actuel incorrect'}), 400
    is_valid, msg = validate_password_strength(new_password)
    if not is_valid:
        return jsonify({'error': msg}), 400
    if new_password == current_password:
        return jsonify({'error': "Le nouveau mot de passe doit être différent"}), 400

    user.password = new_password
    db.session.commit()
    log_activity(user.id, 'password_changed', 'Mot de passe modifié')

    body = f"Bonjour {user.full_name},\n\nVotre mot de passe a été modifié.\n\nL'équipe CommSight"
    send_email(user.email, 'CommSight - Mot de passe modifié', body)
    return jsonify({'message': 'Mot de passe modifié'}), 200

# ============================================
# POSTS
# ============================================

@app.route('/api/posts', methods=['GET'])
@require_auth
def get_posts(user):
    if user.role == 'admin':
        all_posts = Post.query.order_by(Post.created_at.desc()).all()
    else:
        all_posts = Post.query.filter(
            (Post.department == 'all') | (Post.department == user.department)
        ).order_by(Post.created_at.desc()).all()
    return jsonify([p.to_dict() for p in all_posts])

@app.route('/api/posts', methods=['POST'])
@require_admin
def create_post(user):
    data = request.json
    post = Post(
        title      = data.get('title'),
        content    = data.get('content'),
        type       = data.get('type', 'general'),
        department = data.get('department', 'all'),
        attachments= data.get('attachments', []),
        author_id  = user.id,
    )
    db.session.add(post)
    db.session.commit()

    targets = User.query.filter_by(role='employee').all()
    if data.get('department') != 'all':
        targets = [u for u in targets if u.department == data.get('department')]
    for t in targets:
        create_notification(t.id, 'Nouvelle actualité', data.get('title'), 'info')

    log_activity(user.id, 'create_post', f"Publication: {data.get('title')}")
    _refresh_ml_config()
    return jsonify(post.to_dict()), 201

@app.route('/api/posts/<int:post_id>/like', methods=['POST'])
@require_auth
def like_post(user, post_id):
    post = db.session.get(Post, post_id)
    if not post:
        return jsonify({'error': 'Post non trouvé'}), 404
    post.likes += 1
    db.session.commit()
    return jsonify(post.to_dict())

# ============================================
# TÂCHES
# ============================================

@app.route('/api/tasks', methods=['GET'])
@require_auth
def get_tasks(user):
    if user.role == 'admin':
        all_tasks = Task.query.all()
    else:
        all_tasks = Task.query.filter(
            (Task.department == user.department) | (Task.assigned_to_id == user.id)
        ).all()
    return jsonify([t.to_dict() for t in all_tasks])

@app.route('/api/tasks', methods=['POST'])
@require_admin
def create_task(user):
    data = request.json
    deadline = None
    if data.get('deadline'):
        try:
            deadline = datetime.fromisoformat(data['deadline'])
        except ValueError:
            pass

    task = Task(
        code           = generate_task_code(),
        title          = data.get('title'),
        description    = data.get('description', ''),
        priority       = data.get('priority', 'medium'),
        department     = data.get('department'),
        assigned_to_id = data.get('assigned_to_id'),
        created_by_id  = user.id,
        deadline       = deadline,
        attachments    = data.get('attachments', []),
    )
    db.session.add(task)
    db.session.commit()

    if task.assigned_to_id:
        create_notification(task.assigned_to_id, 'Nouvelle tâche assignée', f"{task.code}: {task.title}", 'task')
    else:
        dept_users = User.query.filter_by(department=task.department, role='employee').all()
        for du in dept_users:
            create_notification(du.id, 'Nouvelle tâche département', f"{task.code}: {task.title}", 'task')

    log_activity(user.id, 'create_task', f"Tâche créée: {task.code}")
    _refresh_ml_config()
    return jsonify(task.to_dict()), 201

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
@require_auth
def update_task(user, task_id):
    task = db.session.get(Task, task_id)
    if not task:
        return jsonify({'error': 'Tâche non trouvée'}), 404
    if user.role != 'admin' and task.department != user.department:
        return jsonify({'error': 'Accès refusé'}), 403

    data = request.json
    if 'status' in data:
        old = task.status
        task.status = data['status']
        if data['status'] == 'done':
            task.completed_at = datetime.utcnow()
            create_notification(1, 'Tâche terminée', f"{task.code} complétée par {user.full_name}", 'success')
        log_activity(user.id, 'update_task_status', f"{task.code}: {old} → {data['status']}")

    for field in ('title', 'description', 'priority'):
        if field in data:
            setattr(task, field, data[field])

    if 'deadline' in data and data['deadline']:
        try:
            task.deadline = datetime.fromisoformat(data['deadline'])
        except ValueError:
            pass

    task.updated_at = datetime.utcnow()
    db.session.commit()
    _refresh_ml_config()
    return jsonify(task.to_dict())

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
@require_admin
def delete_task(user, task_id):
    task = db.session.get(Task, task_id)
    if not task:
        return jsonify({'error': 'Tâche non trouvée'}), 404
    log_activity(user.id, 'delete_task', f"Tâche supprimée: {task.code}")
    db.session.delete(task)
    db.session.commit()
    _refresh_ml_config()
    return jsonify({'message': 'Tâche supprimée'}), 200

# ============================================
# CONGÉS
# ============================================

@app.route('/api/leaves', methods=['GET'])
@require_auth
def get_leaves(user):
    if user.role == 'admin':
        all_leaves = Leave.query.all()
    else:
        all_leaves = Leave.query.filter_by(employee_id=user.id).all()
    return jsonify([l.to_dict() for l in all_leaves])

@app.route('/api/leaves', methods=['POST'])
@require_auth
def create_leave(user):
    data  = request.json
    leave = Leave(
        type        = data.get('type'),
        start_date  = data.get('start_date'),
        end_date    = data.get('end_date'),
        reason      = data.get('reason', ''),
        employee_id = user.id,
    )
    db.session.add(leave)
    db.session.commit()
    create_notification(1, 'Nouvelle demande de congé', f"{user.full_name} a demandé un congé", 'leave')
    log_activity(user.id, 'create_leave', f"Demande congé: {data.get('start_date')} → {data.get('end_date')}")
    _refresh_ml_config()
    return jsonify(leave.to_dict()), 201

@app.route('/api/leaves/<int:leave_id>/review', methods=['PUT'])
@require_admin
def review_leave(user, leave_id):
    leave = db.session.get(Leave, leave_id)
    if not leave:
        return jsonify({'error': 'Congé non trouvé'}), 404
    data = request.json
    leave.status      = data.get('status')
    leave.reviewed_by = user.full_name
    leave.reviewed_at = datetime.utcnow()
    db.session.commit()
    create_notification(
        leave.employee_id,
        f"Congé {leave.status}",
        f"Votre demande de congé a été {leave.status}",
        'success' if leave.status == 'approved' else 'error'
    )
    log_activity(user.id, 'review_leave', f"Congé {leave.status}: {leave.employee.full_name}")
    return jsonify(leave.to_dict())

# ============================================
# NOTIFICATIONS
# ============================================

@app.route('/api/notifications', methods=['GET'])
@require_auth
def get_notifications(user):
    notifs = Notification.query.filter_by(user_id=user.id)\
               .order_by(Notification.created_at.desc()).all()
    return jsonify([n.to_dict() for n in notifs])

@app.route('/api/notifications/<int:notif_id>/read', methods=['PUT'])
@require_auth
def mark_notification_read(user, notif_id):
    notif = Notification.query.filter_by(id=notif_id, user_id=user.id).first()
    if not notif:
        return jsonify({'error': 'Notification non trouvée'}), 404
    notif.read = True
    db.session.commit()
    return jsonify(notif.to_dict())

# ============================================
# DASHBOARD ADMIN
# ============================================

@app.route('/api/dashboard/stats', methods=['GET'])
@require_admin
def get_dashboard_stats(user):
    tasks_all = Task.query.all()
    now       = datetime.utcnow()
    stats = {
        'users': {
            'total': User.query.filter_by(role='employee').count(),
            'by_department': {}
        },
        'tasks': {
            'total': len(tasks_all),
            'by_status': {
                'todo':        sum(1 for t in tasks_all if t.status == 'todo'),
                'in_progress': sum(1 for t in tasks_all if t.status == 'in_progress'),
                'done':        sum(1 for t in tasks_all if t.status == 'done'),
                'blocked':     sum(1 for t in tasks_all if t.status == 'blocked'),
            },
            'by_priority': {
                'low':    sum(1 for t in tasks_all if t.priority == 'low'),
                'medium': sum(1 for t in tasks_all if t.priority == 'medium'),
                'high':   sum(1 for t in tasks_all if t.priority == 'high'),
                'urgent': sum(1 for t in tasks_all if t.priority == 'urgent'),
            },
            'overdue': sum(1 for t in tasks_all if t.deadline and t.deadline < now and t.status != 'done'),
        },
        'posts': {
            'total':     Post.query.count(),
            'this_week': Post.query.filter(Post.created_at > now - timedelta(days=7)).count(),
        },
        'leaves': {
            'pending':  Leave.query.filter_by(status='pending').count(),
            'approved': Leave.query.filter_by(status='approved').count(),
            'rejected': Leave.query.filter_by(status='rejected').count(),
        },
        'activity':     [a.to_dict() for a in Activity.query.order_by(Activity.timestamp.desc()).limit(10).all()],
        'ml_available': ML_AVAILABLE,
    }
    depts = db.session.query(User.department).filter_by(role='employee').distinct().all()
    for (dept,) in depts:
        stats['users']['by_department'][dept] = User.query.filter_by(department=dept).count()
    return jsonify(stats)

# ============================================
# ANALYTICS CLASSIQUES
# ============================================

@app.route('/api/analytics/communication', methods=['GET'])
@require_admin
def get_communication_analytics(user):
    isolated = []
    for emp in User.query.filter_by(role='employee').all():
        count = Task.query.filter_by(assigned_to_id=emp.id).count()
        if count < 2:
            isolated.append({'id': emp.id, 'name': emp.full_name, 'department': emp.department, 'task_count': count})
    return jsonify({'isolated_employees': isolated, 'communication_score': 75,
                    'note': 'ML avancé: GET /api/ml/collaboration/network'})

@app.route('/api/analytics/performance', methods=['GET'])
@require_admin
def get_performance_analytics(user):
    depts = db.session.query(User.department).filter_by(role='employee').distinct().all()
    departments = []
    for (dept,) in depts:
        all_t     = Task.query.filter_by(department=dept).all()
        completed = sum(1 for t in all_t if t.status == 'done')
        total     = len(all_t)
        score     = round((completed / total * 100), 1) if total > 0 else 0
        departments.append({
            'name': dept, 'score': score,
            'tasks_completed': completed, 'tasks_total': total,
            'employees_count': User.query.filter_by(department=dept).count()
        })
    return jsonify({'departments': departments, 'overall_score': 80,
                    'note': 'ML avancé: GET /api/ml/full-analysis'})

@app.route('/api/analytics/sentiment', methods=['GET'])
@require_admin
def get_sentiment_analytics_legacy(user):
    all_fb = Feedback.query.all()
    pos    = sum(1 for f in all_fb if f.sentiment == 'positive')
    neg    = sum(1 for f in all_fb if f.sentiment == 'negative')
    total  = len(all_fb)
    score  = 50 + ((pos - neg) / total * 50) if total > 0 else 50
    return jsonify({
        'sentiment_score': round(score, 1),
        'positive_count':  pos,
        'negative_count':  neg,
        'neutral_count':   sum(1 for f in all_fb if f.sentiment == 'neutral'),
        'total_feedbacks': total,
        'ml_endpoint':     '/api/ml/sentiment/batch-analysis',
    })

# ============================================
# MESSAGERIE
# ============================================

@app.route('/api/messages/conversations', methods=['GET'])
@require_auth
def get_conversations(user):
    convs = Conversation.query.filter(
        Conversation.participants.any(id=user.id)
    ).all()
    result = []
    for conv in convs:
        msgs     = Message.query.filter_by(conversation_id=conv.id).order_by(Message.created_at.desc()).all()
        last_msg = msgs[0].to_dict() if msgs else None
        unread   = sum(1 for m in msgs if m.sender_id != user.id and not m.read)
        other    = next((p for p in conv.participants if p.id != user.id), None)
        result.append({
            **conv.to_dict(),
            'name':         conv.name if conv.type == 'group' else (other.full_name if other else 'Inconnu'),
            'avatar':       other.avatar if other and conv.type == 'private' else None,
            'department':   other.department if other and conv.type == 'private' else conv.department,
            'last_message': last_msg,
            'unread_count': unread,
        })
    result.sort(key=lambda x: x['last_message']['created_at'] if x['last_message'] else x['created_at'], reverse=True)
    return jsonify(result)

@app.route('/api/messages/conversation/<int:user_id>', methods=['GET', 'POST'])
@require_auth
def get_or_create_conversation(user, user_id):
    other_user = db.session.get(User, user_id)
    if not other_user:
        return jsonify({'error': 'Utilisateur non trouvé'}), 404

    existing = Conversation.query.filter_by(type='private').filter(
        Conversation.participants.any(id=user.id)
    ).filter(
        Conversation.participants.any(id=user_id)
    ).first()

    if existing:
        return jsonify(existing.to_dict())

    conv = Conversation(type='private')
    conv.participants.append(user)
    conv.participants.append(other_user)
    db.session.add(conv)
    db.session.commit()
    _refresh_ml_config()
    return jsonify(conv.to_dict()), 201

@app.route('/api/messages/<int:conversation_id>', methods=['GET'])
@require_auth
def get_messages_route(user, conversation_id):
    conv = db.session.get(Conversation, conversation_id)
    if not conv:
        return jsonify({'error': 'Conversation non trouvée'}), 404
    if user not in conv.participants:
        return jsonify({'error': 'Accès refusé'}), 403
    msgs = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.created_at).all()
    return jsonify([m.to_dict() for m in msgs])

@app.route('/api/messages/send', methods=['POST'])
@require_auth
def send_message_route(user):
    data            = request.json
    conversation_id = data.get('conversation_id')
    content         = data.get('content', '')

    conv = db.session.get(Conversation, conversation_id)
    if not conv:
        return jsonify({'error': 'Conversation non trouvée'}), 404
    if user not in conv.participants:
        return jsonify({'error': 'Accès refusé'}), 403

    msg = Message(
        conversation_id = conversation_id,
        sender_id       = user.id,
        content         = content,
        attachment      = data.get('attachment'),
    )
    db.session.add(msg)
    db.session.commit()

    for participant in conv.participants:
        if participant.id != user.id:
            snippet = content[:50] + '...' if len(content) > 50 else content
            create_notification(participant.id, 'Nouveau message', f"{user.full_name}: {snippet}", 'info')

    log_activity(user.id, 'send_message', f"Message conv {conversation_id}")
    _refresh_ml_config()
    return jsonify(msg.to_dict()), 201

@app.route('/api/messages/<int:message_id>/read', methods=['PUT'])
@require_auth
def mark_message_read(user, message_id):
    msg = db.session.get(Message, message_id)
    if not msg:
        return jsonify({'error': 'Message non trouvé'}), 404
    conv = db.session.get(Conversation, msg.conversation_id)
    if not conv or user not in conv.participants:
        return jsonify({'error': 'Accès refusé'}), 403
    msg.read = True
    db.session.commit()
    return jsonify(msg.to_dict())

@app.route('/api/messages/groups', methods=['GET'])
@require_auth
def get_groups(user):
    if user.role == 'admin':
        depts = [d for (d,) in db.session.query(User.department).filter_by(role='employee').distinct().all()]
    else:
        depts = [user.department]

    groups = []
    for dept in depts:
        conv = Conversation.query.filter_by(type='group', department=dept).first()
        if not conv:
            conv = Conversation(type='group', name=f"Groupe {dept}", department=dept)
            dept_users = User.query.filter(
                (User.department == dept) | (User.role == 'admin')
            ).all()
            conv.participants.extend(dept_users)
            db.session.add(conv)
            db.session.commit()
        groups.append(conv.to_dict())
    return jsonify(groups)

@app.route('/api/users/search', methods=['GET'])
@require_auth
def search_users(user):
    q    = request.args.get('q', '').lower()
    base = User.query.filter(User.id != user.id)
    if q:
        base = base.filter(
            db.or_(
                User.full_name.ilike(f'%{q}%'),
                User.email.ilike(f'%{q}%'),
                User.department.ilike(f'%{q}%'),
            )
        )
    return jsonify([{
        'id': u.id, 'full_name': u.full_name, 'email': u.email,
        'department': u.department, 'position': u.position, 'avatar': u.avatar
    } for u in base.all()])

@app.route('/api/users/by-department', methods=['GET'])
@require_admin
def get_users_by_department(user):
    departments = [
        'Informatique', 'Ressources Humaines', 'Finance',
        'Marketing', 'Commercial', 'Production', 'Logistique'
    ]
    result = {}
    for dept in departments:
        employees = User.query.filter_by(department=dept).all()
        result[dept] = [u.to_dict() for u in employees]
    return jsonify(result)

# ============================================
# UPLOAD
# ============================================

@app.route('/api/upload', methods=['POST'])
@require_auth
def upload_file(user):
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Fichier invalide ou type non autorisé'}), 400
    filename = secure_filename(f"{datetime.utcnow().timestamp()}_{file.filename}")
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return jsonify({'filename': filename, 'url': f'/uploads/{filename}'}), 201

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ============================================
# UTILISATEURS
# ============================================

@app.route('/api/users', methods=['GET'])
@require_admin
def get_users(user):
    return jsonify([u.to_dict() for u in User.query.filter_by(role='employee').all()])

# ── Export des données personnelles ─────────────────────────
@app.route('/api/users/export-data', methods=['GET'])
@require_auth
def export_user_data(user):
    tasks            = Task.query.filter_by(assigned_to_id=user.id).all()
    leaves           = Leave.query.filter_by(employee_id=user.id).all()
    messages         = Message.query.filter_by(sender_id=user.id).all()
    notifs           = Notification.query.filter_by(user_id=user.id).all()
    activities       = Activity.query.filter_by(user_id=user.id)\
                           .order_by(Activity.timestamp.desc()).all()
    feedbacks        = Feedback.query.filter_by(author_id=user.id).all()
    survey_responses = SurveyResponse.query.filter_by(user_id=user.id).all()
    documents        = Document.query.filter_by(uploaded_by=user.id).all()
    login_history    = LoginHistory.query.filter_by(user_id=user.id)\
                           .order_by(LoginHistory.timestamp.desc()).all()

    export = {
        'export_date':        datetime.utcnow().isoformat(),
        'user':               user.to_dict(),
        'tasks':              [t.to_dict() for t in tasks],
        'leaves':             [l.to_dict() for l in leaves],
        'messages_sent':      [m.to_dict() for m in messages],
        'notifications':      [n.to_dict() for n in notifs],
        'activities':         [a.to_dict() for a in activities],
        'feedbacks':          [f.to_dict() for f in feedbacks],
        'survey_responses':   [r.to_dict() for r in survey_responses],
        'documents_uploaded': [d.to_dict() for d in documents],
        'login_history':      [h.to_dict() for h in login_history],
    }

    log_activity(user.id, 'export_data', 'Export des données personnelles')

    return Response(
        json.dumps(export, ensure_ascii=False, indent=2),
        mimetype='application/json',
        headers={'Content-Disposition': f'attachment; filename=mes-donnees-{user.id}.json'}
    )

# ── Mise à jour du profil ────────────────────────────────────
@app.route('/api/users/<int:user_id>', methods=['PUT'])
@require_auth
def update_user(user, user_id):
    if user.role != 'admin' and user.id != user_id:
        return jsonify({'error': 'Accès refusé'}), 403

    target = db.session.get(User, user_id)
    if not target:
        return jsonify({'error': 'Utilisateur non trouvé'}), 404

    data = request.json

    if 'full_name' in data and data['full_name'].strip():
        target.full_name = data['full_name'].strip()
    if 'phone' in data:
        target.phone = data['phone'].strip()
    if 'position' in data and data['position'].strip():
        target.position = data['position'].strip()

    if user.role == 'admin':
        if 'department' in data and data['department'].strip():
            target.department = data['department'].strip()
        if 'email' in data and '@' in data['email']:
            existing = User.query.filter_by(email=data['email']).first()
            if existing and existing.id != user_id:
                return jsonify({'error': 'Cet email est déjà utilisé'}), 400
            target.email = data['email'].strip()
        if 'role' in data and data['role'] in ('admin', 'employee', 'chef_departement'):
            target.role = data['role']

    db.session.commit()
    log_activity(user.id, 'update_profile', f"Profil mis à jour: {target.full_name}")
    _refresh_ml_config()
    return jsonify(target.to_dict())

@app.route('/api/users/<int:user_id>/set-chef', methods=['PUT'])
@require_admin
def set_chef(user, user_id):
    target = db.session.get(User, user_id)
    if not target:
        return jsonify({'error': 'Utilisateur non trouvé'}), 404

    data   = request.json
    action = data.get('action')

    if action == 'promote':
        old_chef = User.query.filter_by(
            department=target.department,
            role='chef_departement'
        ).first()
        if old_chef and old_chef.id != user_id:
            old_chef.role = 'employee'
            db.session.commit()
        target.role = 'chef_departement'
        log_activity(user.id, 'promote_chef', f"{target.full_name} nommé chef de {target.department}")
    elif action == 'revoke':
        if target.role != 'chef_departement':
            return jsonify({'error': "Cet utilisateur n'est pas chef de département"}), 400
        target.role = 'employee'
        log_activity(user.id, 'revoke_chef', f"{target.full_name} révoqué de {target.department}")
    else:
        return jsonify({'error': 'Action invalide. Utilisez promote ou revoke'}), 400

    db.session.commit()
    _refresh_ml_config()
    return jsonify(target.to_dict())

# ── Suppression de compte ────────────────────────────────────
@app.route('/api/users/<int:user_id>', methods=['DELETE'])
@require_auth
def delete_user(user, user_id):
    if user.role != 'admin' and user.id != user_id:
        return jsonify({'error': 'Accès refusé'}), 403

    target = db.session.get(User, user_id)
    if not target:
        return jsonify({'error': 'Utilisateur non trouvé'}), 404

    if target.role == 'admin':
        admin_count = User.query.filter_by(role='admin').count()
        if admin_count <= 1:
            return jsonify({'error': 'Impossible de supprimer le dernier administrateur'}), 400

    if user.id == user_id:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        sessions.pop(token, None)
        if 'sessions' in app.config:
            app.config['sessions'].pop(token, None)

    full_name = target.full_name

    Notification.query.filter_by(user_id=user_id).delete()
    Activity.query.filter_by(user_id=user_id).delete()
    LoginHistory.query.filter_by(user_id=user_id).delete()
    Feedback.query.filter_by(author_id=user_id).update({'author_id': None})
    SurveyResponse.query.filter_by(user_id=user_id).update({'user_id': None})

    db.session.delete(target)
    db.session.commit()

    _refresh_ml_config()
    print(f"🗑️  Compte supprimé: {full_name} (id={user_id})")
    return jsonify({'message': 'Compte supprimé avec succès'}), 200

# ============================================
# ENQUÊTES & FEEDBACKS
# ============================================

def _analyze_sentiment_basic(text):
    tl  = text.lower()
    pos = sum(1 for w in ['bon','bien','excellent','super','génial','parfait','merci','content','satisfait','heureux'] if w in tl)
    neg = sum(1 for w in ['mauvais','mal','problème','jamais','rien','difficile','impossible','mécontent','insatisfait'] if w in tl)
    if pos > neg: return 'positive'
    if neg > pos: return 'negative'
    return 'neutral'

@app.route('/api/surveys', methods=['GET'])
@require_auth
def get_surveys(user):
    surveys = Survey.query.all()
    result  = []
    for s in surveys:
        if s.target_department not in ('all', user.department) and user.role != 'admin':
            continue
        d = s.to_dict()
        d['has_responded']  = SurveyResponse.query.filter_by(survey_id=s.id, user_id=user.id).count() > 0
        d['response_count'] = SurveyResponse.query.filter_by(survey_id=s.id).count()
        result.append(d)
    result.sort(key=lambda x: x['created_at'], reverse=True)
    return jsonify(result)

@app.route('/api/surveys', methods=['POST'])
@require_admin
def create_survey(user):
    data     = request.json
    deadline = None
    if data.get('deadline'):
        try:
            deadline = datetime.fromisoformat(data['deadline'])
        except ValueError:
            pass
    survey = Survey(
        title             = data.get('title'),
        description       = data.get('description', ''),
        target_department = data.get('target_department', 'all'),
        anonymous         = data.get('anonymous', False),
        questions         = data.get('questions', []),
        deadline          = deadline,
        show_results      = data.get('show_results', False),
        created_by_id     = user.id,
    )
    db.session.add(survey)
    db.session.commit()

    targets = User.query.filter_by(role='employee').all()
    if survey.target_department != 'all':
        targets = [u for u in targets if u.department == survey.target_department]
    for t in targets:
        create_notification(t.id, 'Nouvelle enquête', f"{survey.title} - Votre avis compte !", 'info')

    log_activity(user.id, 'create_survey', f"Enquête créée: {survey.title}")
    return jsonify(survey.to_dict()), 201

@app.route('/api/surveys/<int:survey_id>', methods=['GET'])
@require_auth
def get_survey_detail(user, survey_id):
    survey = db.session.get(Survey, survey_id)
    if not survey:
        return jsonify({'error': 'Enquête non trouvée'}), 404
    if survey.target_department not in ('all', user.department) and user.role != 'admin':
        return jsonify({'error': 'Accès refusé'}), 403
    return jsonify(survey.to_dict())

@app.route('/api/surveys/<int:survey_id>/respond', methods=['POST'])
@require_auth
def respond_to_survey(user, survey_id):
    survey = db.session.get(Survey, survey_id)
    if not survey:
        return jsonify({'error': 'Enquête non trouvée'}), 404
    if SurveyResponse.query.filter_by(survey_id=survey_id, user_id=user.id).count():
        return jsonify({'error': 'Vous avez déjà répondu'}), 400

    data     = request.json
    response = SurveyResponse(
        survey_id  = survey_id,
        user_id    = user.id if not survey.anonymous else None,
        department = user.department,
        answers    = data.get('answers', []),
    )
    db.session.add(response)
    db.session.commit()
    log_activity(user.id, 'respond_survey', f"Réponse enquête: {survey.title}")
    return jsonify({'message': 'Réponse enregistrée', 'response': response.to_dict()}), 201

@app.route('/api/surveys/<int:survey_id>/results', methods=['GET'])
@require_admin
def get_survey_results(user, survey_id):
    survey = db.session.get(Survey, survey_id)
    if not survey:
        return jsonify({'error': 'Enquête non trouvée'}), 404

    responses   = SurveyResponse.query.filter_by(survey_id=survey_id).all()
    total_emp_q = User.query.filter_by(role='employee')
    if survey.target_department != 'all':
        total_emp_q = total_emp_q.filter_by(department=survey.target_department)
    total_emp = total_emp_q.count()

    results = {
        'survey':             survey.to_dict(),
        'total_responses':    len(responses),
        'response_rate':      round(len(responses) / total_emp * 100, 1) if total_emp else 0,
        'questions_analysis': []
    }
    for i, question in enumerate(survey.questions or []):
        qa = {'question': question, 'responses': [r.answers[i] for r in responses if i < len(r.answers or [])]}
        if question['type'] in ('single_choice', 'multiple_choice'):
            counts = {}
            for ans in qa['responses']:
                for opt in (ans if isinstance(ans, list) else [ans]):
                    counts[opt] = counts.get(opt, 0) + 1
            qa['option_counts'] = counts
        elif question['type'] == 'scale':
            nums = [int(r) for r in qa['responses'] if r]
            if nums:
                qa['average'] = round(sum(nums) / len(nums), 2)
                qa['min']     = min(nums)
                qa['max']     = max(nums)
        elif question['type'] == 'text':
            qa['text_responses'] = qa['responses']
        results['questions_analysis'].append(qa)

    return jsonify(results)

@app.route('/api/surveys/<int:survey_id>', methods=['DELETE'])
@require_admin
def delete_survey(user, survey_id):
    survey = db.session.get(Survey, survey_id)
    if not survey:
        return jsonify({'error': 'Enquête non trouvée'}), 404
    log_activity(user.id, 'delete_survey', f"Enquête supprimée: {survey.title}")
    db.session.delete(survey)
    db.session.commit()
    return jsonify({'message': 'Enquête supprimée'}), 200

@app.route('/api/feedbacks', methods=['GET'])
@require_admin
def get_feedbacks(user):
    all_fb = Feedback.query.order_by(Feedback.created_at.desc()).all()
    return jsonify([f.to_dict() for f in all_fb])

@app.route('/api/feedbacks', methods=['POST'])
@require_auth
def create_feedback(user):
    data     = request.json
    content  = data.get('content', '')
    category = data.get('category', 'suggestion')

    sentiment = _analyze_sentiment_basic(content)
    if ML_AVAILABLE:
        try:
            from ml_engine import orchestrator
            r = orchestrator.sentiment_analyzer.analyze_text(content, category=None)
            sentiment = r.get('label', sentiment)
        except Exception:
            pass

    fb = Feedback(
        category  = category,
        title     = data.get('title'),
        content   = content,
        department= user.department,
        sentiment = sentiment,
        author_id = user.id,
    )
    db.session.add(fb)
    db.session.commit()
    create_notification(1, 'Nouveau feedback', f"{fb.category.capitalize()}: {fb.title}", 'info')
    log_activity(user.id, 'create_feedback', f"Feedback soumis: {fb.category}")
    _refresh_ml_config()
    return jsonify({'message': 'Feedback enregistré anonymement', 'feedback_id': fb.id}), 201

@app.route('/api/feedbacks/<int:feedback_id>/respond', methods=['POST'])
@require_admin
def respond_to_feedback(user, feedback_id):
    fb = db.session.get(Feedback, feedback_id)
    if not fb:
        return jsonify({'error': 'Feedback non trouvé'}), 404
    data      = request.json
    responses = fb.responses or []
    responses.append({'admin': user.full_name, 'message': data.get('message'), 'created_at': datetime.utcnow().isoformat()})
    fb.responses = responses
    fb.status    = data.get('status', 'in_progress')
    db.session.commit()
    log_activity(user.id, 'respond_feedback', f"Réponse feedback #{feedback_id}")
    return jsonify(fb.to_dict())

@app.route('/api/feedbacks/stats', methods=['GET'])
@require_admin
def get_feedback_stats(user):
    all_fb = Feedback.query.all()
    stats  = {
        'total':         len(all_fb),
        'by_category':   {c: sum(1 for f in all_fb if f.category == c) for c in ['suggestion','probleme','idee','autre']},
        'by_status':     {s: sum(1 for f in all_fb if f.status == s)   for s in ['new','in_progress','resolved','archived']},
        'by_sentiment':  {s: sum(1 for f in all_fb if f.sentiment == s) for s in ['positive','neutral','negative']},
        'by_department': {},
        'ml_endpoint':   '/api/ml/sentiment/batch-analysis',
    }
    for fb in all_fb:
        d = fb.department or 'N/A'
        stats['by_department'][d] = stats['by_department'].get(d, 0) + 1
    return jsonify(stats)

# ============================================
# ARCHIVAGE & GED
# ============================================

@app.route('/api/archives/folders', methods=['GET'])
@require_auth
def get_folders(user):
    roots = DocumentFolder.query.filter_by(parent_id=None).all()
    return jsonify([f.to_dict(with_children=True) for f in roots])

@app.route('/api/archives/folders', methods=['POST'])
@require_admin
def create_folder(user):
    data   = request.json
    folder = DocumentFolder(
        name       = data.get('name'),
        parent_id  = data.get('parent_id'),
        icon       = data.get('icon', 'folder'),
        color      = data.get('color', 'blue'),
        created_by = user.id,
    )
    db.session.add(folder)
    db.session.commit()
    log_activity(user.id, 'create_folder', f"Dossier créé: {folder.name}")
    return jsonify(folder.to_dict()), 201

@app.route('/api/archives/documents', methods=['GET'])
@require_auth
def get_documents(user):
    folder_id = request.args.get('folder_id', type=int)
    search    = request.args.get('search', '').lower()
    file_type = request.args.get('type')

    q = Document.query
    if folder_id:
        q = q.filter_by(folder_id=folder_id)
    if search:
        q = q.filter(db.or_(Document.name.ilike(f'%{search}%'), Document.description.ilike(f'%{search}%')))
    if file_type:
        q = q.filter(Document.mime_type.like(f'{file_type}%'))

    docs   = q.order_by(Document.uploaded_at.desc()).all()
    result = []
    for doc in docs:
        d = doc.to_dict()
        if doc.uploader:
            d['uploader'] = {'id': doc.uploader.id, 'full_name': doc.uploader.full_name, 'department': doc.uploader.department}
        result.append(d)
    return jsonify(result)

@app.route('/api/archives/documents', methods=['POST'])
@require_auth
def upload_document(user):
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    file        = request.files['file']
    folder_id   = request.form.get('folder_id', type=int)
    description = request.form.get('description', '')
    tags        = request.form.get('tags', '')

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Fichier invalide'}), 400

    original = secure_filename(file.filename)
    unique   = f"{datetime.utcnow().timestamp()}_{original}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique)
    file.save(filepath)

    doc = Document(
        name        = original,
        filename    = unique,
        folder_id   = folder_id,
        description = description,
        tags        = tags.split(',') if tags else [],
        mime_type   = mimetypes.guess_type(original)[0] or 'application/octet-stream',
        size        = os.path.getsize(filepath),
        uploaded_by = user.id,
    )
    db.session.add(doc)
    db.session.commit()
    log_activity(user.id, 'upload_document', f"Document uploadé: {original}")
    return jsonify(doc.to_dict()), 201

@app.route('/api/archives/documents/<int:doc_id>', methods=['GET'])
@require_auth
def get_document_detail(user, doc_id):
    doc = db.session.get(Document, doc_id)
    if not doc:
        return jsonify({'error': 'Document non trouvé'}), 404
    return jsonify(doc.to_dict())

@app.route('/api/archives/documents/<int:doc_id>/download', methods=['GET'])
@require_auth
def download_document(user, doc_id):
    doc = db.session.get(Document, doc_id)
    if not doc:
        return jsonify({'error': 'Document non trouvé'}), 404
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], doc.filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'Fichier introuvable sur le serveur'}), 404
    doc.downloads += 1
    db.session.commit()
    log_activity(user.id, 'download_document', f"Téléchargement: {doc.name}")
    return send_from_directory(app.config['UPLOAD_FOLDER'], doc.filename, as_attachment=True, download_name=doc.name)

@app.route('/api/archives/documents/<int:doc_id>', methods=['PUT'])
@require_auth
def update_document(user, doc_id):
    doc = db.session.get(Document, doc_id)
    if not doc:
        return jsonify({'error': 'Document non trouvé'}), 404
    if doc.uploaded_by != user.id and user.role != 'admin':
        return jsonify({'error': 'Permission refusée'}), 403
    data = request.json
    for field in ('name', 'description', 'tags', 'folder_id'):
        if field in data:
            setattr(doc, field, data[field])
    db.session.commit()
    log_activity(user.id, 'update_document', f"Document modifié: {doc.name}")
    return jsonify(doc.to_dict())

@app.route('/api/archives/documents/<int:doc_id>', methods=['DELETE'])
@require_auth
def delete_document(user, doc_id):
    doc = db.session.get(Document, doc_id)
    if not doc:
        return jsonify({'error': 'Document non trouvé'}), 404
    if doc.uploaded_by != user.id and user.role != 'admin':
        return jsonify({'error': 'Permission refusée'}), 403
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], doc.filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    log_activity(user.id, 'delete_document', f"Document supprimé: {doc.name}")
    db.session.delete(doc)
    db.session.commit()
    return jsonify({'message': 'Document supprimé'}), 200

@app.route('/api/archives/documents/<int:doc_id>/share', methods=['POST'])
@require_auth
def share_document(user, doc_id):
    doc = db.session.get(Document, doc_id)
    if not doc:
        return jsonify({'error': 'Document non trouvé'}), 404
    data   = request.json
    shared = doc.shared_with or []
    for uid in data.get('user_ids', []):
        if uid not in shared:
            shared.append(uid)
            target = db.session.get(User, uid)
            if target:
                create_notification(uid, 'Document partagé', f"{user.full_name} a partagé '{doc.name}' avec vous", 'info')
    doc.shared_with = shared
    db.session.commit()
    log_activity(user.id, 'share_document', f"Document partagé: {doc.name}")
    return jsonify(doc.to_dict())

@app.route('/api/archives/stats', methods=['GET'])
@require_auth
def get_archives_stats(user):
    docs  = Document.query.all()
    stats = {
        'total_documents': len(docs),
        'total_size':      sum(d.size or 0 for d in docs),
        'total_downloads': sum(d.downloads for d in docs),
        'by_type':         {},
        'by_folder':       {},
        'recent_uploads':  [],
    }
    for doc in docs:
        t = (doc.mime_type or 'unknown').split('/')[0]
        stats['by_type'][t] = stats['by_type'].get(t, 0) + 1
    for folder in DocumentFolder.query.all():
        count = Document.query.filter_by(folder_id=folder.id).count()
        if count:
            stats['by_folder'][folder.name] = count
    for doc in Document.query.order_by(Document.uploaded_at.desc()).limit(5).all():
        up = doc.uploader
        stats['recent_uploads'].append({
            'id':           doc.id,
            'name':         doc.name,
            'uploaded_at':  doc.uploaded_at.isoformat(),
            'uploader':     up.full_name if up else 'Inconnu',
        })
    return jsonify(stats)

# ============================================
# ÉVALUATIONS (Chef de département)
# ============================================

@app.route('/api/evaluations', methods=['GET'])
@require_auth
def get_evaluations(user):
    if not CHEF_MODELS_AVAILABLE:
        return jsonify([])
    if user.role == 'admin':
        evals = Evaluation.query.all()
    elif user.role == 'chef_departement':
        evals = Evaluation.query.filter_by(department=user.department).all()
    else:
        evals = Evaluation.query.filter_by(employee_id=user.id).all()
    return jsonify([e.to_dict() for e in evals])

@app.route('/api/evaluations', methods=['POST'])
@require_auth
def create_evaluation(user):
    if not CHEF_MODELS_AVAILABLE:
        return jsonify({'error': 'Fonctionnalité non disponible'}), 501
    if user.role not in ('admin', 'chef_departement'):
        return jsonify({'error': 'Accès refusé'}), 403
    data = request.json
    ev = Evaluation(
        employee_id  = data.get('employee_id'),
        evaluator_id = user.id,
        department   = user.department,
        period       = data.get('period'),
        scores       = data.get('scores', {}),
        comment      = data.get('comment', ''),
        global_score = data.get('global_score', 0),
    )
    db.session.add(ev)
    db.session.commit()
    create_notification(ev.employee_id, 'Nouvelle évaluation', 'Vous avez reçu une évaluation de votre chef de département', 'info')
    log_activity(user.id, 'create_evaluation', f"Évaluation créée pour employé #{ev.employee_id}")
    return jsonify(ev.to_dict()), 201

@app.route('/api/evaluations/<int:eval_id>', methods=['PUT'])
@require_auth
def update_evaluation(user, eval_id):
    if not CHEF_MODELS_AVAILABLE:
        return jsonify({'error': 'Fonctionnalité non disponible'}), 501
    if user.role not in ('admin', 'chef_departement'):
        return jsonify({'error': 'Accès refusé'}), 403
    ev = db.session.get(Evaluation, eval_id)
    if not ev:
        return jsonify({'error': 'Évaluation non trouvée'}), 404
    data = request.json
    for field in ('scores', 'comment', 'global_score', 'period'):
        if field in data:
            setattr(ev, field, data[field])
    db.session.commit()
    return jsonify(ev.to_dict())

@app.route('/api/evaluations/<int:eval_id>', methods=['DELETE'])
@require_auth
def delete_evaluation(user, eval_id):
    if not CHEF_MODELS_AVAILABLE:
        return jsonify({'error': 'Fonctionnalité non disponible'}), 501
    if user.role not in ('admin', 'chef_departement'):
        return jsonify({'error': 'Accès refusé'}), 403
    ev = db.session.get(Evaluation, eval_id)
    if not ev:
        return jsonify({'error': 'Évaluation non trouvée'}), 404
    db.session.delete(ev)
    db.session.commit()
    return jsonify({'message': 'Évaluation supprimée'})

# ============================================
# PRIMES (Chef de département)
# ============================================

@app.route('/api/primes', methods=['GET'])
@require_auth
def get_primes(user):
    if not CHEF_MODELS_AVAILABLE:
        return jsonify([])
    if user.role == 'admin':
        primes = Prime.query.all()
    elif user.role == 'chef_departement':
        primes = Prime.query.filter_by(department=user.department).all()
    else:
        primes = Prime.query.filter_by(employee_id=user.id).all()
    return jsonify([p.to_dict() for p in primes])

@app.route('/api/primes', methods=['POST'])
@require_auth
def create_prime(user):
    if not CHEF_MODELS_AVAILABLE:
        return jsonify({'error': 'Fonctionnalité non disponible'}), 501
    if user.role not in ('admin', 'chef_departement'):
        return jsonify({'error': 'Accès refusé'}), 403
    data  = request.json
    prime = Prime(
        employee_id   = data.get('employee_id'),
        attributed_by = user.id,
        department    = user.department,
        type          = data.get('type', 'performance'),
        amount        = data.get('amount', 0),
        period        = data.get('period'),
        reason        = data.get('reason', ''),
        status        = 'pending',
    )
    db.session.add(prime)
    db.session.commit()
    create_notification(prime.employee_id, 'Prime attribuée', f"Une prime de {prime.amount} DA vous a été attribuée", 'success')
    log_activity(user.id, 'create_prime', f"Prime {prime.amount} DA pour employé #{prime.employee_id}")
    return jsonify(prime.to_dict()), 201

@app.route('/api/primes/<int:prime_id>', methods=['PUT'])
@require_auth
def update_prime(user, prime_id):
    if not CHEF_MODELS_AVAILABLE:
        return jsonify({'error': 'Fonctionnalité non disponible'}), 501
    if user.role not in ('admin', 'chef_departement'):
        return jsonify({'error': 'Accès refusé'}), 403
    prime = db.session.get(Prime, prime_id)
    if not prime:
        return jsonify({'error': 'Prime non trouvée'}), 404
    data = request.json
    for field in ('amount', 'reason', 'status', 'period', 'type'):
        if field in data:
            setattr(prime, field, data[field])
    db.session.commit()
    return jsonify(prime.to_dict())

@app.route('/api/primes/<int:prime_id>', methods=['DELETE'])
@require_auth
def delete_prime(user, prime_id):
    if not CHEF_MODELS_AVAILABLE:
        return jsonify({'error': 'Fonctionnalité non disponible'}), 501
    if user.role not in ('admin', 'chef_departement'):
        return jsonify({'error': 'Accès refusé'}), 403
    prime = db.session.get(Prime, prime_id)
    if not prime:
        return jsonify({'error': 'Prime non trouvée'}), 404
    db.session.delete(prime)
    db.session.commit()
    return jsonify({'message': 'Prime supprimée'})

# ============================================
# RECRUTEMENT (Chef de département)
# ============================================

@app.route('/api/recrutement', methods=['GET'])
@require_auth
def get_recrutement(user):
    if not CHEF_MODELS_AVAILABLE:
        return jsonify([])
    if user.role == 'admin':
        postes = PosteOuvert.query.all()
    elif user.role == 'chef_departement':
        postes = PosteOuvert.query.filter_by(department=user.department).all()
    else:
        return jsonify({'error': 'Accès refusé'}), 403
    return jsonify([p.to_dict() for p in postes])

@app.route('/api/recrutement', methods=['POST'])
@require_auth
def create_poste(user):
    if not CHEF_MODELS_AVAILABLE:
        return jsonify({'error': 'Fonctionnalité non disponible'}), 501
    if user.role not in ('admin', 'chef_departement'):
        return jsonify({'error': 'Accès refusé'}), 403
    data  = request.json
    poste = PosteOuvert(
        title        = data.get('title'),
        department   = user.department,
        description  = data.get('description', ''),
        requirements = data.get('requirements', ''),
        type_contrat = data.get('type_contrat', 'CDI'),
        nb_postes    = data.get('nb_postes', 1),
        created_by   = user.id,
        status       = 'open',
    )
    db.session.add(poste)
    db.session.commit()
    log_activity(user.id, 'create_poste', f"Poste ouvert: {poste.title}")
    return jsonify(poste.to_dict()), 201

@app.route('/api/recrutement/<int:poste_id>', methods=['PUT'])
@require_auth
def update_poste(user, poste_id):
    if not CHEF_MODELS_AVAILABLE:
        return jsonify({'error': 'Fonctionnalité non disponible'}), 501
    if user.role not in ('admin', 'chef_departement'):
        return jsonify({'error': 'Accès refusé'}), 403
    poste = db.session.get(PosteOuvert, poste_id)
    if not poste:
        return jsonify({'error': 'Poste non trouvé'}), 404
    data = request.json
    for field in ('title', 'description', 'requirements', 'type_contrat', 'nb_postes', 'status'):
        if field in data:
            setattr(poste, field, data[field])
    db.session.commit()
    return jsonify(poste.to_dict())

@app.route('/api/recrutement/<int:poste_id>', methods=['DELETE'])
@require_auth
def delete_poste(user, poste_id):
    if not CHEF_MODELS_AVAILABLE:
        return jsonify({'error': 'Fonctionnalité non disponible'}), 501
    if user.role not in ('admin', 'chef_departement'):
        return jsonify({'error': 'Accès refusé'}), 403
    poste = db.session.get(PosteOuvert, poste_id)
    if not poste:
        return jsonify({'error': 'Poste non trouvé'}), 404
    db.session.delete(poste)
    db.session.commit()
    return jsonify({'message': 'Poste supprimé'})

@app.route('/api/recrutement/<int:poste_id>/candidats', methods=['GET'])
@require_auth
def get_candidats(user, poste_id):
    if not CHEF_MODELS_AVAILABLE:
        return jsonify([])
    if user.role not in ('admin', 'chef_departement'):
        return jsonify({'error': 'Accès refusé'}), 403
    candidats = Candidat.query.filter_by(poste_id=poste_id).all()
    return jsonify([c.to_dict() for c in candidats])

@app.route('/api/recrutement/<int:poste_id>/candidats', methods=['POST'])
@require_auth
def add_candidat(user, poste_id):
    if not CHEF_MODELS_AVAILABLE:
        return jsonify({'error': 'Fonctionnalité non disponible'}), 501
    if user.role not in ('admin', 'chef_departement'):
        return jsonify({'error': 'Accès refusé'}), 403
    data     = request.json
    candidat = Candidat(
        poste_id  = poste_id,
        full_name = data.get('full_name'),
        email     = data.get('email'),
        phone     = data.get('phone', ''),
        cv_url    = data.get('cv_url', ''),
        note      = data.get('note', ''),
        status    = 'nouveau',
        added_by  = user.id,
    )
    db.session.add(candidat)
    db.session.commit()
    return jsonify(candidat.to_dict()), 201

@app.route('/api/recrutement/candidats/<int:candidat_id>', methods=['PUT'])
@require_auth
def update_candidat(user, candidat_id):
    if not CHEF_MODELS_AVAILABLE:
        return jsonify({'error': 'Fonctionnalité non disponible'}), 501
    if user.role not in ('admin', 'chef_departement'):
        return jsonify({'error': 'Accès refusé'}), 403
    candidat = db.session.get(Candidat, candidat_id)
    if not candidat:
        return jsonify({'error': 'Candidat non trouvé'}), 404
    data = request.json
    for field in ('status', 'note', 'phone', 'email'):
        if field in data:
            setattr(candidat, field, data[field])
    db.session.commit()
    return jsonify(candidat.to_dict())

@app.route('/api/recrutement/candidats/<int:candidat_id>', methods=['DELETE'])
@require_auth
def delete_candidat(user, candidat_id):
    if not CHEF_MODELS_AVAILABLE:
        return jsonify({'error': 'Fonctionnalité non disponible'}), 501
    if user.role not in ('admin', 'chef_departement'):
        return jsonify({'error': 'Accès refusé'}), 403
    candidat = db.session.get(Candidat, candidat_id)
    if not candidat:
        return jsonify({'error': 'Candidat non trouvé'}), 404
    db.session.delete(candidat)
    db.session.commit()
    return jsonify({'message': 'Candidat supprimé'})

# ============================================
# ML — INITIALISATION
# ============================================

def init_ml_engine():
    if not ML_AVAILABLE:
        return
    try:
        _refresh_ml_config()
        from ml_engine import orchestrator
        result = orchestrator.initialize(
            app.config['users'], app.config['tasks'], app.config['messages'],
            app.config['leaves'], app.config['activities'], app.config['feedbacks'],
            app.config['conversations']
        )
        print(f"🤖 ML initialisé: {result.get('n_employees', 0)} employés")
    except Exception as e:
        print(f"⚠️  ML partiel: {e}")

# ============================================
# SEED DATABASE
# ============================================

def seed_database():
    if User.query.count() > 0:
        return

    print("📦 Initialisation de la base de données...")

    admin = User(
        email          = 'admin@commsight.com',
        password       = 'Admin@2025!',
        full_name      = 'Administrateur Principal',
        role           = 'admin',
        department     = 'Direction',
        position       = 'Directeur Général',
        phone          = '+213 555 123 456',
        email_verified = True,
    )
    db.session.add(admin)

    folders = [
        DocumentFolder(id=1, name='Documents RH',            parent_id=None, icon='users',       color='blue'),
        DocumentFolder(id=2, name='Contrats',                 parent_id=1,    icon='file-text',   color='green'),
        DocumentFolder(id=3, name='Fiches de paie',           parent_id=1,    icon='credit-card', color='purple'),
        DocumentFolder(id=4, name='Documents Administratifs', parent_id=None, icon='briefcase',   color='orange'),
        DocumentFolder(id=5, name='Factures',                 parent_id=4,    icon='file',        color='red'),
        DocumentFolder(id=6, name='Rapports',                 parent_id=None, icon='bar-chart',   color='cyan'),
        DocumentFolder(id=7, name='Projets',                  parent_id=None, icon='folder',      color='indigo'),
    ]
    db.session.add_all(folders)
    db.session.commit()
    print("✅ Base de données initialisée")

# ============================================
# DÉMARRAGE
# ============================================

with app.app_context():
    db.create_all()
    seed_database()
    init_ml_engine()
    print("✅ Application initialisée")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n{'='*70}")
    print(" COMMSIGHT — Backend + SQLAlchemy + Moteur ML")
    print(f"{'='*70}")
    print(f"\n📍 Serveur:    http://localhost:{port}")
    print(f"💾 Base:       {app.config['SQLALCHEMY_DATABASE_URI']}")
    print(f"\n👤 Admin:      admin@commsight.com / Admin@2025!")
    print(f"🤖 ML:         {'✅ disponible' if ML_AVAILABLE else '❌ pip install scikit-learn xgboost shap nltk prophet pandas'}")
    print(f"\n{'='*70}\n")
    app.run(debug=False, host='0.0.0.0', port=port)