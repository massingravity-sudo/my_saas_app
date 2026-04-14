# ============================================================
# MODELS.PY — Modèles SQLAlchemy pour CommSight
# Base de données : PostgreSQL (prod) / SQLite (dev local)
# ============================================================

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

# ═══════════════════════════════════════════════════════════
# USERS
# ═══════════════════════════════════════════════════════════

class User(db.Model):
    __tablename__ = 'users'

    id              = db.Column(db.Integer, primary_key=True)
    email           = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password        = db.Column(db.String(255), nullable=False)
    full_name       = db.Column(db.String(255), nullable=False)
    role            = db.Column(db.String(50), nullable=False, default='employee')  # admin | employee
    department      = db.Column(db.String(100))
    position        = db.Column(db.String(100), default='Employé')
    phone           = db.Column(db.String(50))
    avatar          = db.Column(db.String(500))
    email_verified  = db.Column(db.Boolean, default=False)
    last_login      = db.Column(db.DateTime)
    login_attempts  = db.Column(db.Integer, default=0)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)

    # Relations
    tasks_assigned  = db.relationship('Task', foreign_keys='Task.assigned_to_id', backref='assigned_to', lazy='dynamic')
    tasks_created   = db.relationship('Task', foreign_keys='Task.created_by_id', backref='created_by', lazy='dynamic')
    leaves          = db.relationship('Leave', backref='employee', lazy='dynamic')
    notifications   = db.relationship('Notification', backref='user', lazy='dynamic')
    activities      = db.relationship('Activity', backref='user', lazy='dynamic')
    documents       = db.relationship('Document', backref='uploader', lazy='dynamic')
    feedbacks       = db.relationship('Feedback', backref='author', lazy='dynamic')

    def to_dict(self, include_password=False):
        data = {
            'id': self.id,
            'email': self.email,
            'full_name': self.full_name,
            'role': self.role,
            'department': self.department,
            'position': self.position,
            'phone': self.phone,
            'avatar': self.avatar,
            'email_verified': self.email_verified,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'login_attempts': self.login_attempts,
            'created_at': self.created_at.isoformat(),
        }
        if include_password:
            data['password'] = self.password
        return data


# ═══════════════════════════════════════════════════════════
# LOGIN HISTORY
# ═══════════════════════════════════════════════════════════

class LoginHistory(db.Model):
    __tablename__ = 'login_history'

    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    email       = db.Column(db.String(255))
    ip_address  = db.Column(db.String(100))
    user_agent  = db.Column(db.String(500))
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'email': self.email,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'timestamp': self.timestamp.isoformat(),
        }


# ═══════════════════════════════════════════════════════════
# POSTS
# ═══════════════════════════════════════════════════════════

class Post(db.Model):
    __tablename__ = 'posts'

    id          = db.Column(db.Integer, primary_key=True)
    title       = db.Column(db.String(500), nullable=False)
    content     = db.Column(db.Text, nullable=False)
    type        = db.Column(db.String(50), default='general')
    department  = db.Column(db.String(100), default='all')
    attachments = db.Column(db.JSON, default=list)
    likes       = db.Column(db.Integer, default=0)
    comments    = db.Column(db.JSON, default=list)
    author_id   = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    author = db.relationship('User', backref='posts')

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'type': self.type,
            'department': self.department,
            'attachments': self.attachments or [],
            'likes': self.likes,
            'comments': self.comments or [],
            'author': {
                'id': self.author.id,
                'full_name': self.author.full_name,
                'role': self.author.role,
            },
            'created_at': self.created_at.isoformat(),
        }


# ═══════════════════════════════════════════════════════════
# TASKS
# ═══════════════════════════════════════════════════════════

class Task(db.Model):
    __tablename__ = 'tasks'

    id              = db.Column(db.Integer, primary_key=True)
    code            = db.Column(db.String(50), unique=True, nullable=False)
    title           = db.Column(db.String(500), nullable=False)
    description     = db.Column(db.Text, default='')
    status          = db.Column(db.String(50), default='todo')   # todo | in_progress | done | blocked
    priority        = db.Column(db.String(50), default='medium') # low | medium | high | urgent
    department      = db.Column(db.String(100))
    assigned_to_id  = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_by_id   = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    deadline        = db.Column(db.DateTime, nullable=True)
    attachments     = db.Column(db.JSON, default=list)
    workflow_step   = db.Column(db.Integer, default=1)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at      = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at    = db.Column(db.DateTime, nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'code': self.code,
            'title': self.title,
            'description': self.description,
            'status': self.status,
            'priority': self.priority,
            'department': self.department,
            'assigned_to_id': self.assigned_to_id,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'attachments': self.attachments or [],
            'workflow_step': self.workflow_step,
            'created_by': {
                'id': self.created_by.id,
                'full_name': self.created_by.full_name,
            } if self.created_by else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }


# ═══════════════════════════════════════════════════════════
# LEAVES (CONGÉS)
# ═══════════════════════════════════════════════════════════

class Leave(db.Model):
    __tablename__ = 'leaves'

    id              = db.Column(db.Integer, primary_key=True)
    type            = db.Column(db.String(100), nullable=False)
    start_date      = db.Column(db.String(50), nullable=False)
    end_date        = db.Column(db.String(50), nullable=False)
    reason          = db.Column(db.Text, default='')
    status          = db.Column(db.String(50), default='pending')  # pending | approved | rejected
    employee_id     = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    reviewed_by     = db.Column(db.String(255))
    reviewed_at     = db.Column(db.DateTime, nullable=True)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        emp = self.employee
        return {
            'id': self.id,
            'type': self.type,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'reason': self.reason,
            'status': self.status,
            'employee': {
                'id': emp.id,
                'full_name': emp.full_name,
                'department': emp.department,
            } if emp else None,
            'reviewed_by': self.reviewed_by,
            'reviewed_at': self.reviewed_at.isoformat() if self.reviewed_at else None,
            'created_at': self.created_at.isoformat(),
        }


# ═══════════════════════════════════════════════════════════
# NOTIFICATIONS
# ═══════════════════════════════════════════════════════════

class Notification(db.Model):
    __tablename__ = 'notifications'

    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title       = db.Column(db.String(255), nullable=False)
    message     = db.Column(db.Text, nullable=False)
    type        = db.Column(db.String(50), default='info')
    read        = db.Column(db.Boolean, default=False)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'message': self.message,
            'type': self.type,
            'read': self.read,
            'created_at': self.created_at.isoformat(),
        }


# ═══════════════════════════════════════════════════════════
# ACTIVITIES
# ═══════════════════════════════════════════════════════════

class Activity(db.Model):
    __tablename__ = 'activities'

    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    action      = db.Column(db.String(100), nullable=False)
    details     = db.Column(db.Text)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'action': self.action,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
        }


# ═══════════════════════════════════════════════════════════
# MESSAGING
# ═══════════════════════════════════════════════════════════

# Table de jointure conversation ↔ participants
conversation_participants = db.Table(
    'conversation_participants',
    db.Column('conversation_id', db.Integer, db.ForeignKey('conversations.id'), primary_key=True),
    db.Column('user_id',         db.Integer, db.ForeignKey('users.id'),          primary_key=True),
)

class Conversation(db.Model):
    __tablename__ = 'conversations'

    id          = db.Column(db.Integer, primary_key=True)
    type        = db.Column(db.String(50), default='private')  # private | group
    name        = db.Column(db.String(255))
    department  = db.Column(db.String(100))
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    participants = db.relationship('User', secondary=conversation_participants, lazy='subquery')
    messages     = db.relationship('Message', backref='conversation', lazy='dynamic',
                                   cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type,
            'name': self.name,
            'department': self.department,
            'participants': [u.id for u in self.participants],
            'created_at': self.created_at.isoformat(),
        }


class Message(db.Model):
    __tablename__ = 'messages'

    id              = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversations.id'), nullable=False)
    sender_id       = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    content         = db.Column(db.Text, nullable=False)
    attachment      = db.Column(db.String(500))
    read            = db.Column(db.Boolean, default=False)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)

    sender = db.relationship('User', backref='sent_messages')

    def to_dict(self):
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'sender_id': self.sender_id,
            'content': self.content,
            'attachment': self.attachment,
            'read': self.read,
            'created_at': self.created_at.isoformat(),
            'sender': {
                'id': self.sender.id,
                'full_name': self.sender.full_name,
                'avatar': self.sender.avatar,
                'department': self.sender.department,
            } if self.sender else None,
        }


# ═══════════════════════════════════════════════════════════
# SURVEYS & FEEDBACKS
# ═══════════════════════════════════════════════════════════

class Survey(db.Model):
    __tablename__ = 'surveys'

    id                  = db.Column(db.Integer, primary_key=True)
    title               = db.Column(db.String(500), nullable=False)
    description         = db.Column(db.Text, default='')
    target_department   = db.Column(db.String(100), default='all')
    anonymous           = db.Column(db.Boolean, default=False)
    questions           = db.Column(db.JSON, default=list)
    deadline            = db.Column(db.DateTime, nullable=True)
    show_results        = db.Column(db.Boolean, default=False)
    status              = db.Column(db.String(50), default='active')
    created_by_id       = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at          = db.Column(db.DateTime, default=datetime.utcnow)

    creator   = db.relationship('User', backref='surveys')
    responses = db.relationship('SurveyResponse', backref='survey', lazy='dynamic',
                                cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'target_department': self.target_department,
            'anonymous': self.anonymous,
            'questions': self.questions or [],
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'show_results': self.show_results,
            'status': self.status,
            'created_by': {
                'id': self.creator.id,
                'full_name': self.creator.full_name,
            } if self.creator else None,
            'created_at': self.created_at.isoformat(),
        }


class SurveyResponse(db.Model):
    __tablename__ = 'survey_responses'

    id          = db.Column(db.Integer, primary_key=True)
    survey_id   = db.Column(db.Integer, db.ForeignKey('surveys.id'), nullable=False)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # NULL si anonyme
    department  = db.Column(db.String(100))
    answers     = db.Column(db.JSON, default=list)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'survey_id': self.survey_id,
            'user_id': self.user_id,
            'department': self.department,
            'answers': self.answers or [],
            'created_at': self.created_at.isoformat(),
        }


class Feedback(db.Model):
    __tablename__ = 'feedbacks'

    id          = db.Column(db.Integer, primary_key=True)
    category    = db.Column(db.String(100), default='suggestion')  # suggestion | probleme | idee | autre
    title       = db.Column(db.String(500), nullable=False)
    content     = db.Column(db.Text, nullable=False)
    department  = db.Column(db.String(100))
    anonymous   = db.Column(db.Boolean, default=True)
    status      = db.Column(db.String(50), default='new')   # new | in_progress | resolved | archived
    priority    = db.Column(db.String(50), default='medium')
    sentiment   = db.Column(db.String(50), default='neutral')
    responses   = db.Column(db.JSON, default=list)
    author_id   = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'category': self.category,
            'title': self.title,
            'content': self.content,
            'department': self.department,
            'anonymous': self.anonymous,
            'status': self.status,
            'priority': self.priority,
            'sentiment': self.sentiment,
            'responses': self.responses or [],
            'created_at': self.created_at.isoformat(),
        }


# ═══════════════════════════════════════════════════════════
# DOCUMENTS & ARCHIVAGE
# ═══════════════════════════════════════════════════════════

class DocumentFolder(db.Model):
    __tablename__ = 'document_folders'

    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(255), nullable=False)
    parent_id   = db.Column(db.Integer, db.ForeignKey('document_folders.id'), nullable=True)
    icon        = db.Column(db.String(50), default='folder')
    color       = db.Column(db.String(50), default='blue')
    created_by  = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    children    = db.relationship('DocumentFolder', backref=db.backref('parent', remote_side=[id]))
    documents   = db.relationship('Document', backref='folder', lazy='dynamic')

    def to_dict(self, with_children=False):
        data = {
            'id': self.id,
            'name': self.name,
            'parent_id': self.parent_id,
            'icon': self.icon,
            'color': self.color,
            'document_count': self.documents.count(),
        }
        if with_children:
            data['children'] = [c.to_dict(with_children=True) for c in self.children]
        return data


class Document(db.Model):
    __tablename__ = 'documents'

    id              = db.Column(db.Integer, primary_key=True)
    name            = db.Column(db.String(500), nullable=False)
    filename        = db.Column(db.String(500), nullable=False)  # nom sur disque
    folder_id       = db.Column(db.Integer, db.ForeignKey('document_folders.id'), nullable=True)
    description     = db.Column(db.Text, default='')
    tags            = db.Column(db.JSON, default=list)
    mime_type       = db.Column(db.String(255))
    size            = db.Column(db.BigInteger, default=0)
    uploaded_by     = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    version         = db.Column(db.Integer, default=1)
    downloads       = db.Column(db.Integer, default=0)
    shared_with     = db.Column(db.JSON, default=list)
    uploaded_at     = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'filename': self.filename,
            'folder_id': self.folder_id,
            'description': self.description,
            'tags': self.tags or [],
            'mime_type': self.mime_type,
            'size': self.size,
            'uploaded_by': self.uploaded_by,
            'version': self.version,
            'downloads': self.downloads,
            'shared_with': self.shared_with or [],
            'uploaded_at': self.uploaded_at.isoformat(),
        }



class Evaluation(db.Model):
    __tablename__ = 'evaluations'
    id           = db.Column(db.Integer, primary_key=True)
    employee_id  = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    evaluator_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    department   = db.Column(db.String(100))
    period       = db.Column(db.String(50))   # ex: "Q1 2026"
    scores       = db.Column(db.JSON, default={})
    comment      = db.Column(db.Text, default='')
    global_score = db.Column(db.Float, default=0)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

    employee  = db.relationship('User', foreign_keys=[employee_id])
    evaluator = db.relationship('User', foreign_keys=[evaluator_id])

    def to_dict(self):
        return {
            'id': self.id,
            'employee_id': self.employee_id,
            'employee': {'id': self.employee.id, 'full_name': self.employee.full_name, 'position': self.employee.position} if self.employee else None,
            'evaluator': {'id': self.evaluator.id, 'full_name': self.evaluator.full_name} if self.evaluator else None,
            'department': self.department,
            'period': self.period,
            'scores': self.scores,
            'comment': self.comment,
            'global_score': self.global_score,
            'created_at': self.created_at.isoformat(),
        }

class Prime(db.Model):
    __tablename__ = 'primes'
    id            = db.Column(db.Integer, primary_key=True)
    employee_id   = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    attributed_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    department    = db.Column(db.String(100))
    type          = db.Column(db.String(50), default='performance')
    amount        = db.Column(db.Float, default=0)
    period        = db.Column(db.String(50))
    reason        = db.Column(db.Text, default='')
    status        = db.Column(db.String(30), default='pending')  # pending/approved/paid
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    employee  = db.relationship('User', foreign_keys=[employee_id])
    attributor= db.relationship('User', foreign_keys=[attributed_by])

    def to_dict(self):
        return {
            'id': self.id,
            'employee_id': self.employee_id,
            'employee': {'id': self.employee.id, 'full_name': self.employee.full_name, 'position': self.employee.position} if self.employee else None,
            'attributed_by': {'id': self.attributor.id, 'full_name': self.attributor.full_name} if self.attributor else None,
            'department': self.department,
            'type': self.type,
            'amount': self.amount,
            'period': self.period,
            'reason': self.reason,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
        }

class PosteOuvert(db.Model):
    __tablename__ = 'postes_ouverts'
    id            = db.Column(db.Integer, primary_key=True)
    title         = db.Column(db.String(200), nullable=False)
    department    = db.Column(db.String(100))
    description   = db.Column(db.Text, default='')
    requirements  = db.Column(db.Text, default='')
    type_contrat  = db.Column(db.String(50), default='CDI')
    nb_postes     = db.Column(db.Integer, default=1)
    status        = db.Column(db.String(30), default='open')  # open/closed/paused
    created_by    = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    candidats     = db.relationship('Candidat', backref='poste', lazy=True, cascade='all, delete-orphan')
    creator       = db.relationship('User', foreign_keys=[created_by])

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'department': self.department,
            'description': self.description,
            'requirements': self.requirements,
            'type_contrat': self.type_contrat,
            'nb_postes': self.nb_postes,
            'status': self.status,
            'created_by': {'id': self.creator.id, 'full_name': self.creator.full_name} if self.creator else None,
            'nb_candidats': len(self.candidats),
            'created_at': self.created_at.isoformat(),
        }

class Candidat(db.Model):
    __tablename__ = 'candidats'
    id         = db.Column(db.Integer, primary_key=True)
    poste_id   = db.Column(db.Integer, db.ForeignKey('postes_ouverts.id'), nullable=False)
    full_name  = db.Column(db.String(200), nullable=False)
    email      = db.Column(db.String(200))
    phone      = db.Column(db.String(50), default='')
    cv_url     = db.Column(db.String(500), default='')
    note       = db.Column(db.Text, default='')
    status     = db.Column(db.String(30), default='nouveau')  # nouveau/entretien/retenu/rejeté
    added_by   = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'poste_id': self.poste_id,
            'full_name': self.full_name,
            'email': self.email,
            'phone': self.phone,
            'cv_url': self.cv_url,
            'note': self.note,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
        }