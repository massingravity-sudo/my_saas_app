from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import jwt
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)

# ── CORS ─────────────────────────────────────────────────────────────────────
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
    }
})

# ── Configuration ─────────────────────────────────────────────────────────────
SECRET_KEY = 'votre_cle_secrete_super_secure_2024'
app.config['SECRET_KEY'] = SECRET_KEY

EMAIL_CONFIG = {
    'smtp_server':    'smtp.gmail.com',
    'smtp_port':      587,
    'sender_email':   'benslimanefaiz7@gmail.com',
    'sender_password':'yaeylorbnbjsztsj',
    'sender_name':    'CommSight'
}

# ═════════════════════════════════════════════════════════════════════════════
# BASES DE DONNÉES EN MÉMOIRE
# ═════════════════════════════════════════════════════════════════════════════
organizations   = [];  org_id_counter   = 1
users           = [];  user_id_counter  = 1
otp_codes       = {}
leaves          = [];  leave_id_counter = 1
tasks           = [];  task_id_counter  = 1
messages        = [];  message_id_counter = 1
conversations   = [];  conversation_id_counter = 1
posts           = [];  post_id_counter  = 1
surveys         = [];  survey_id_counter = 1
survey_responses= [];  survey_response_id_counter = 1
feedbacks       = [];  feedback_id_counter = 1
notifications   = [];  notification_id_counter = 1
activity_logs   = [];  activity_log_id_counter = 1
evaluations     = [];  evaluation_id_counter = 1
bonuses         = [];  bonus_id_counter = 1
absences        = [];  absence_id_counter = 1
trainings       = [];  training_id_counter = 1
training_enrollments = []; training_enrollment_id_counter = 1


def _sync_app_config():
    """
    Synchronise les listes globales vers app.config.
    Les blueprints ML lisent depuis current_app.config — cette fonction
    garantit que les données sont toujours à jour.
    """
    app.config['users']         = users
    app.config['tasks']         = tasks
    app.config['messages']      = messages
    app.config['leaves']        = leaves
    app.config['activities']    = activity_logs
    app.config['feedbacks']     = feedbacks
    app.config['conversations'] = conversations

# Sync initiale
_sync_app_config()


# ═════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═════════════════════════════════════════════════════════════════════════════

def generate_token(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(days=7)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
    # Compatible PyJWT 1.x (bytes) ET 2.x (str)
    return token if isinstance(token, str) else token.decode('utf-8')


def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload['user_id']
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def get_user_from_token(token):
    user_id = verify_token(token)
    if user_id:
        return next((u for u in users if u['id'] == user_id), None)
    return None


def send_email(recipient_email, subject, body):
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From']    = f"{EMAIL_CONFIG['sender_name']} <{EMAIL_CONFIG['sender_email']}>"
        msg['To']      = recipient_email
        html_body = f"""
        <html><body style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
          <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                      padding:30px;text-align:center;">
            <h1 style="color:white;margin:0;">CommSight</h1>
          </div>
          <div style="padding:30px;background:#f7fafc;">{body}</div>
          <div style="padding:20px;text-align:center;color:#718096;font-size:12px;">
            <p>© 2026 CommSight — Tous droits réservés</p>
          </div>
        </body></html>
        """
        msg.attach(MIMEText(html_body, 'html'))
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Erreur envoi email: {e}")
        return False


def generate_otp():
    return ''.join([str(secrets.randbelow(10)) for _ in range(6)])


def create_notification(user_id, title, message, notification_type='info'):
    global notification_id_counter
    notif = {
        'id':         notification_id_counter,
        'user_id':    user_id,
        'title':      title,
        'message':    message,
        'type':       notification_type,
        'read':       False,
        'created_at': datetime.now().isoformat()
    }
    notifications.append(notif)
    notification_id_counter += 1
    return notif


def log_activity(user_id, action_type, description):
    global activity_log_id_counter
    activity_logs.append({
        'id':          activity_log_id_counter,
        'user_id':     user_id,
        'action_type': action_type,
        'description': description,
        'timestamp':   datetime.now().isoformat()
    })
    activity_log_id_counter += 1
    _sync_app_config()


# ═════════════════════════════════════════════════════════════════════════════
# DÉCORATEURS
# ═════════════════════════════════════════════════════════════════════════════

def require_auth(f):
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        user  = get_user_from_token(token)
        if not user:
            return jsonify({'error': 'Non authentifié'}), 401
        return f(user, *args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


def require_admin(f):
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        user  = get_user_from_token(token)
        if not user or user['role'] != 'admin':
            return jsonify({'error': 'Accès refusé — droits admin requis'}), 403
        return f(user, *args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


def require_manager_or_admin(f):
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        user  = get_user_from_token(token)
        if not user or user['role'] not in ['admin', 'manager']:
            return jsonify({'error': 'Accès refusé — droits manager ou admin requis'}), 403
        return f(user, *args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — AUTH
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/register', methods=['POST', 'OPTIONS'])
def register():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    global org_id_counter, user_id_counter

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Corps de requête JSON manquant'}), 400

    required = {
        'organization_name':     "Nom de l'organisation requis",
        'organization_industry': "Secteur d'activité requis",
        'organization_size':     "Taille de l'organisation requise",
        'full_name':             "Nom complet requis",
        'email':                 "Email requis",
        'password':              "Mot de passe requis",
    }
    for field, msg in required.items():
        if not str(data.get(field, '')).strip():
            return jsonify({'error': msg}), 400

    email = data['email'].strip().lower()
    if '@' not in email:
        return jsonify({'error': 'Format email invalide'}), 400
    if any(u['email'] == email for u in users):
        return jsonify({'error': 'Cet email est déjà utilisé'}), 400
    if len(data['password']) < 6:
        return jsonify({'error': 'Le mot de passe doit contenir au moins 6 caractères'}), 400

    # Organisation
    organization = {
        'id':       org_id_counter,
        'name':     data['organization_name'].strip(),
        'industry': data['organization_industry'].strip(),
        'size':     data['organization_size'].strip(),
        'country':  data.get('organization_country', 'Algérie').strip(),
        'created_at': datetime.now().isoformat(),
        'settings': {
            'features': {
                'surveys': True, 'feedbacks': True, 'analytics': True,
                'ml_analytics': True, 'two_factor_auth': False,
                'employee_registration': False
            },
            'security': {
                'password_min_length': 6, 'session_timeout': 30,
                'max_login_attempts': 5, 'password_expiry_days': 90
            }
        }
    }
    organizations.append(organization)
    org_id_counter += 1

    # Utilisateur admin
    user = {
        'id':              user_id_counter,
        'organization_id': organization['id'],
        'full_name':       data['full_name'].strip(),
        'email':           email,
        'password':        data['password'],
        'phone':           data.get('phone', ''),
        'position':        data.get('position', 'Administrateur'),
        'department':      data.get('department', 'Direction'),
        'role':            'admin',
        'created_at':      datetime.now().isoformat(),
        'last_login':      None
    }
    users.append(user)
    user_id_counter += 1

    _sync_app_config()

    token = generate_token(user['id'])

    send_email(
        user['email'], 'Bienvenue sur CommSight',
        f"""
        <h2 style="color:#4c51bf;">Bienvenue {user['full_name']} !</h2>
        <p>Votre organisation <strong>{organization['name']}</strong>
           a été créée avec succès.</p>
        <p>Vous pouvez maintenant vous connecter et commencer à utiliser CommSight.</p>
        """
    )

    log_activity(user['id'], 'register', f"Inscription de {user['full_name']}")

    user_data = {k: v for k, v in user.items() if k != 'password'}
    return jsonify({'token': token, 'user': user_data, 'organization': organization}), 201


@app.route('/api/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Corps de requête JSON manquant'}), 400

    email    = data.get('email', '').strip().lower()
    password = data.get('password', '')
    if not email or not password:
        return jsonify({'error': 'Email et mot de passe requis'}), 400

    user = next((u for u in users if u['email'] == email), None)
    if not user or user['password'] != password:
        return jsonify({'error': 'Email ou mot de passe incorrect'}), 401

    user['last_login'] = datetime.now().isoformat()
    token = generate_token(user['id'])
    log_activity(user['id'], 'login', f"Connexion de {user['full_name']}")

    user_data = {k: v for k, v in user.items() if k != 'password'}
    return jsonify({'token': token, 'user': user_data})


@app.route('/api/forgot-password', methods=['POST', 'OPTIONS'])
def forgot_password():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    data  = request.get_json(silent=True) or {}
    email = data.get('email', '').strip().lower()
    user  = next((u for u in users if u['email'] == email), None)
    if not user:
        return jsonify({'error': 'Aucun compte associé à cet email'}), 404

    otp = generate_otp()
    otp_codes[email] = {'code': otp, 'expires_at': datetime.now() + timedelta(minutes=15)}
    send_email(email, 'Réinitialisation de votre mot de passe',
        f"""
        <h2 style="color:#4c51bf;">Code de vérification</h2>
        <div style="background:#f7fafc;padding:20px;text-align:center;
                    margin:20px 0;border-radius:8px;">
          <h1 style="color:#4c51bf;font-size:36px;letter-spacing:8px;margin:0;">{otp}</h1>
        </div>
        <p>Valable 15 minutes.</p>
        """)
    return jsonify({'message': 'Code envoyé par email'})


@app.route('/api/verify-otp', methods=['POST', 'OPTIONS'])
def verify_otp():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    data  = request.get_json(silent=True) or {}
    email = data.get('email', '').strip().lower()
    code  = data.get('code', '')
    if email not in otp_codes:
        return jsonify({'error': 'Code invalide ou expiré'}), 400
    stored = otp_codes[email]
    if datetime.now() > stored['expires_at']:
        del otp_codes[email]
        return jsonify({'error': 'Code expiré'}), 400
    if stored['code'] != code:
        return jsonify({'error': 'Code incorrect'}), 400
    return jsonify({'message': 'Code vérifié'})


@app.route('/api/reset-password', methods=['POST', 'OPTIONS'])
def reset_password():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    data         = request.get_json(silent=True) or {}
    email        = data.get('email', '').strip().lower()
    code         = data.get('code', '')
    new_password = data.get('new_password', '')
    if email not in otp_codes:
        return jsonify({'error': 'Code invalide ou expiré'}), 400
    stored = otp_codes[email]
    if datetime.now() > stored['expires_at']:
        del otp_codes[email]
        return jsonify({'error': 'Code expiré'}), 400
    if stored['code'] != code:
        return jsonify({'error': 'Code incorrect'}), 400
    user = next((u for u in users if u['email'] == email), None)
    if user:
        user['password'] = new_password
        del otp_codes[email]
        log_activity(user['id'], 'password_reset', 'Réinitialisation du mot de passe')
        return jsonify({'message': 'Mot de passe réinitialisé avec succès'})
    return jsonify({'error': 'Utilisateur non trouvé'}), 404


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — UTILISATEURS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/users', methods=['GET'])
@require_auth
def get_users(user):
    org_users  = [u for u in users if u['organization_id'] == user['organization_id']]
    return jsonify([{k: v for k, v in u.items() if k != 'password'} for u in org_users])


@app.route('/api/users/<int:user_id>', methods=['GET'])
@require_auth
def get_user(user, user_id):
    target = next(
        (u for u in users if u['id'] == user_id and u['organization_id'] == user['organization_id']),
        None
    )
    if not target:
        return jsonify({'error': 'Utilisateur non trouvé'}), 404
    return jsonify({k: v for k, v in target.items() if k != 'password'})


@app.route('/api/users/<int:user_id>', methods=['PUT'])
@require_auth
def update_user(user, user_id):
    if user['id'] != user_id and user['role'] != 'admin':
        return jsonify({'error': 'Non autorisé'}), 403
    target = next((u for u in users if u['id'] == user_id), None)
    if not target:
        return jsonify({'error': 'Utilisateur non trouvé'}), 404
    data = request.get_json(silent=True) or {}
    for field in ['full_name', 'phone', 'position', 'department']:
        if field in data:
            target[field] = data[field]
    _sync_app_config()
    log_activity(user['id'], 'update_profile', f"Profil mis à jour — {target['full_name']}")
    return jsonify({k: v for k, v in target.items() if k != 'password'})


@app.route('/api/users/<int:user_id>', methods=['DELETE'])
@require_admin
def delete_user(user, user_id):
    target = next(
        (u for u in users if u['id'] == user_id and u['organization_id'] == user['organization_id']),
        None
    )
    if not target:
        return jsonify({'error': 'Utilisateur non trouvé'}), 404
    if target['role'] == 'admin':
        return jsonify({'error': 'Impossible de supprimer un administrateur'}), 400
    target.update({
        'full_name':  f"Utilisateur supprimé #{target['id']}",
        'email':      f"deleted_{target['id']}@deleted.com",
        'phone':      '',
        'deleted_at': datetime.now().isoformat()
    })
    _sync_app_config()
    log_activity(user['id'], 'delete_user', f"Utilisateur #{user_id} supprimé")
    return jsonify({'message': 'Utilisateur supprimé'})


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — CONGÉS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/leaves', methods=['GET', 'POST'])
@require_auth
def manage_leaves(user):
    global leave_id_counter
    if request.method == 'GET':
        if user['role'] == 'admin':
            result = [l for l in leaves if l['organization_id'] == user['organization_id']]
        elif user['role'] == 'manager':
            dept_ids = [u['id'] for u in users
                        if u['department'] == user.get('manager_of_department')
                        and u['organization_id'] == user['organization_id']]
            result = [l for l in leaves if l['employee']['id'] in dept_ids]
        else:
            result = [l for l in leaves if l['employee']['id'] == user['id']]
        return jsonify(sorted(result, key=lambda x: x['created_at'], reverse=True))

    data  = request.get_json(silent=True) or {}
    leave = {
        'id':              leave_id_counter,
        'organization_id': user['organization_id'],
        'employee': {'id': user['id'], 'full_name': user['full_name'], 'department': user['department']},
        'type':       data.get('type'),
        'start_date': data.get('start_date'),
        'end_date':   data.get('end_date'),
        'reason':     data.get('reason', ''),
        'status':     'pending',
        'created_at': datetime.now().isoformat(),
        'reviewed_by': None, 'reviewed_at': None
    }
    leaves.append(leave)
    leave_id_counter += 1
    _sync_app_config()
    log_activity(user['id'], 'create_leave', f"Demande de congé: {data.get('type')}")
    return jsonify(leave), 201


@app.route('/api/leaves/<int:leave_id>/review', methods=['PUT'])
@require_manager_or_admin
def review_leave(user, leave_id):
    leave = next((l for l in leaves if l['id'] == leave_id), None)
    if not leave:
        return jsonify({'error': 'Demande non trouvée'}), 404
    data   = request.get_json(silent=True) or {}
    status = data.get('status')
    if status not in ['approved', 'rejected']:
        return jsonify({'error': 'Statut invalide'}), 400
    leave.update({'status': status, 'reviewed_by': user['full_name'],
                  'reviewed_at': datetime.now().isoformat()})
    _sync_app_config()
    create_notification(
        leave['employee']['id'],
        f"Demande de congé {status}",
        f"Votre demande du {leave['start_date']} au {leave['end_date']} a été "
        f"{'approuvée' if status == 'approved' else 'rejetée'}",
        'success' if status == 'approved' else 'warning'
    )
    return jsonify(leave)


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — TASKS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/tasks', methods=['GET', 'POST'])
@require_auth
def manage_tasks(user):
    global task_id_counter
    if request.method == 'GET':
        org_tasks = [t for t in tasks if t['organization_id'] == user['organization_id']]
        if user['role'] != 'admin':
            org_tasks = [t for t in org_tasks
                         if user['id'] in t.get('assigned_to', [])
                         or t.get('created_by') == user['id']]
        return jsonify(sorted(org_tasks, key=lambda x: x['created_at'], reverse=True))

    data = request.get_json(silent=True) or {}
    task = {
        'id':              task_id_counter,
        'organization_id': user['organization_id'],
        'title':           data.get('title', ''),
        'description':     data.get('description', ''),
        'priority':        data.get('priority', 'medium'),
        'status':          'todo',
        'assigned_to':     data.get('assigned_to', []),
        'assigned_to_id':  data.get('assigned_to_id'),
        'department':      data.get('department', user.get('department', '')),
        'deadline':        data.get('deadline') or data.get('due_date'),
        'completed_at':    None,
        'created_by':      user['id'],
        'created_at':      datetime.now().isoformat(),
        'updated_at':      datetime.now().isoformat()
    }
    tasks.append(task)
    task_id_counter += 1
    _sync_app_config()
    return jsonify(task), 201


@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
@require_auth
def update_task(user, task_id):
    task = next((t for t in tasks
                 if t['id'] == task_id and t['organization_id'] == user['organization_id']), None)
    if not task:
        return jsonify({'error': 'Tâche non trouvée'}), 404
    data = request.get_json(silent=True) or {}
    for field in ['title', 'description', 'priority', 'status', 'assigned_to', 'deadline']:
        if field in data:
            task[field] = data[field]
    if data.get('status') == 'done' and not task.get('completed_at'):
        task['completed_at'] = datetime.now().isoformat()
    task['updated_at'] = datetime.now().isoformat()
    _sync_app_config()
    return jsonify(task)


@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
@require_auth
def delete_task(user, task_id):
    task = next((t for t in tasks
                 if t['id'] == task_id and t['organization_id'] == user['organization_id']), None)
    if not task:
        return jsonify({'error': 'Tâche non trouvée'}), 404
    tasks.remove(task)
    _sync_app_config()
    return jsonify({'message': 'Tâche supprimée'})


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — MESSAGES
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/messages', methods=['GET', 'POST'])
@require_auth
def manage_messages(user):
    global message_id_counter
    if request.method == 'GET':
        result = [m for m in messages
                  if m['organization_id'] == user['organization_id']
                  and (m['sender_id'] == user['id'] or user['id'] in m.get('recipients', []))]
        return jsonify(sorted(result, key=lambda x: x['created_at'], reverse=True))

    data = request.get_json(silent=True) or {}
    msg  = {
        'id':              message_id_counter,
        'organization_id': user['organization_id'],
        'sender_id':       user['id'],
        'sender_name':     user['full_name'],
        'recipients':      data.get('recipients', []),
        'subject':         data.get('subject', ''),
        'content':         data.get('content', ''),
        'created_at':      datetime.now().isoformat(),
        'read_by':         []
    }
    messages.append(msg)
    message_id_counter += 1
    _sync_app_config()
    return jsonify(msg), 201


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — POSTS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/posts', methods=['GET', 'POST'])
@require_auth
def manage_posts(user):
    global post_id_counter
    if request.method == 'GET':
        org_posts = [p for p in posts if p['organization_id'] == user['organization_id']]
        return jsonify(sorted(org_posts, key=lambda x: x['created_at'], reverse=True))
    data = request.get_json(silent=True) or {}
    post = {
        'id':              post_id_counter,
        'organization_id': user['organization_id'],
        'title':           data.get('title', ''),
        'content':         data.get('content', ''),
        'author_id':       user['id'],
        'author_name':     user['full_name'],
        'category':        data.get('category', 'general'),
        'likes': [], 'comments': [],
        'created_at':      datetime.now().isoformat(),
    }
    posts.append(post)
    post_id_counter += 1
    return jsonify(post), 201


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — SURVEYS / FEEDBACKS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/surveys', methods=['GET', 'POST'])
@require_auth
def manage_surveys(user):
    global survey_id_counter
    if request.method == 'GET':
        return jsonify([s for s in surveys if s['organization_id'] == user['organization_id']])
    data = request.get_json(silent=True) or {}
    survey = {
        'id':              survey_id_counter,
        'organization_id': user['organization_id'],
        'title':           data.get('title', ''),
        'description':     data.get('description', ''),
        'questions':       data.get('questions', []),
        'status':          'active',
        'created_by':      user['id'],
        'created_at':      datetime.now().isoformat(),
        'closes_at':       data.get('closes_at')
    }
    surveys.append(survey)
    survey_id_counter += 1
    return jsonify(survey), 201


@app.route('/api/feedbacks', methods=['GET', 'POST'])
@require_auth
def manage_feedbacks(user):
    global feedback_id_counter
    if request.method == 'GET':
        return jsonify([f for f in feedbacks if f['organization_id'] == user['organization_id']])
    data = request.get_json(silent=True) or {}
    feedback = {
        'id':                feedback_id_counter,
        'organization_id':   user['organization_id'],
        'title':             data.get('title', ''),
        'content':           data.get('content', ''),
        'category':          data.get('category', 'suggestion'),
        'department':        user.get('department', ''),
        'submitted_by':      user['id'],
        'submitted_by_name': user['full_name'],
        'status':            'pending',
        'sentiment':         'neutral',
        'votes':             0,
        'created_at':        datetime.now().isoformat()
    }
    feedbacks.append(feedback)
    feedback_id_counter += 1
    _sync_app_config()
    return jsonify(feedback), 201


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — NOTIFICATIONS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/notifications', methods=['GET'])
@require_auth
def get_notifications(user):
    user_notifs = [n for n in notifications if n['user_id'] == user['id']]
    return jsonify(sorted(user_notifs, key=lambda x: x['created_at'], reverse=True))


@app.route('/api/notifications/<int:notif_id>/read', methods=['PUT'])
@require_auth
def mark_notification_read(user, notif_id):
    notif = next(
        (n for n in notifications if n['id'] == notif_id and n['user_id'] == user['id']), None
    )
    if not notif:
        return jsonify({'error': 'Notification non trouvée'}), 404
    notif['read'] = True
    return jsonify(notif)


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — ORGANISATION
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/organization/settings', methods=['GET', 'PUT'])
@require_admin
def organization_settings(user):
    org = next((o for o in organizations if o['id'] == user['organization_id']), None)
    if not org:
        return jsonify({'error': 'Organisation non trouvée'}), 404
    if request.method == 'GET':
        return jsonify(org)
    data = request.get_json(silent=True) or {}
    for field in ['name', 'industry', 'size', 'country']:
        if field in data:
            org[field] = data[field]
    return jsonify(org)


@app.route('/api/organization/departments', methods=['GET'])
@require_auth
def get_departments(user):
    org_users   = [u for u in users if u['organization_id'] == user['organization_id']]
    departments = {}
    for u in org_users:
        dept = u.get('department', 'Non assigné')
        if dept not in departments:
            departments[dept] = {'name': dept, 'employee_count': 0, 'manager': None}
        departments[dept]['employee_count'] += 1
        if u.get('role') == 'manager' and u.get('manager_of_department') == dept:
            departments[dept]['manager'] = {'id': u['id'], 'full_name': u['full_name']}
    return jsonify(list(departments.values()))


# ═════════════════════════════════════════════════════════════════════════════
# ROUTE DE TEST
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'CommSight API',
        'version': '2.2',
        'status':  'running',
        'stats': {
            'organizations': len(organizations),
            'users':         len(users),
            'tasks':         len(tasks),
            'leaves':        len(leaves),
            'feedbacks':     len(feedbacks),
        }
    })


# ═════════════════════════════════════════════════════════════════════════════
# ENREGISTREMENT DES BLUEPRINTS ML
# Les blueprints lisent depuis current_app.config — synchronisé par _sync_app_config()
# ═════════════════════════════════════════════════════════════════════════════

try:
    from ml_analytics import ml_bp
    app.register_blueprint(ml_bp)
    print("✅ Blueprint ML Analytics enregistré  → /api/ml/*")
except Exception as e:
    print(f"⚠️  ML Analytics non chargé: {e}")

try:
    from mlops import mlops_bp
    app.register_blueprint(mlops_bp)
    print("✅ Blueprint MLOps enregistré          → /api/mlops/*")
except Exception as e:
    print(f"⚠️  MLOps non chargé: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# LANCEMENT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 55)
    print("🚀 CommSight Backend — port 5000")
    print(f"✅ Email  : {EMAIL_CONFIG['sender_email']}")
    print("✅ CORS   : tous origines (Railway production)")
    print("✅ ML     : chargé si dépendances présentes")
    print("=" * 55)
    app.run(debug=True, port=5000)