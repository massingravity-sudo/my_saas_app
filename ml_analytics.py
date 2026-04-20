# ============================================================
# ML ANALYTICS API — Routes Flask
# CORRECTION : décorateur require_admin branché sur le vrai JWT de app.py
# ============================================================

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
from functools import wraps
import logging

logger = logging.getLogger(__name__)

try:
    from ml_engine import orchestrator
    if orchestrator is None:
        raise ImportError("orchestrator non initialisé")
except Exception as e:
    raise ImportError(f"ML engine indisponible: {e}")

ml_bp = Blueprint('ml_analytics', __name__, url_prefix='/api/ml')


# ═════════════════════════════════════════════════════════════════════════════
# DÉCORATEUR — branché sur le vrai système JWT de app.py
# ═════════════════════════════════════════════════════════════════════════════

def _get_user_from_request():
    """
    Récupère l'utilisateur depuis le token JWT de la requête.
    Utilise la logique de verify_token définie dans app.py.
    """
    import jwt as pyjwt
    token = request.headers.get('Authorization', '').replace('Bearer ', '').strip()
    if not token:
        return None
    try:
        secret = current_app.config.get('SECRET_KEY', '')
        payload = pyjwt.decode(token, secret, algorithms=['HS256'])
        user_id = payload.get('user_id')
        users   = current_app.config.get('users', [])
        return next((u for u in users if u['id'] == user_id), None)
    except Exception:
        return None


def require_admin(f):
    """Vérifie que l'utilisateur est authentifié ET admin."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = _get_user_from_request()
        if not user:
            return jsonify({'error': 'Non authentifié'}), 401
        if user.get('role') != 'admin':
            return jsonify({'error': 'Accès refusé — droits admin requis'}), 403
        return f(user, *args, **kwargs)
    return decorated


# ═════════════════════════════════════════════════════════════════════════════
# HELPER — Lecture des données depuis app.config (synchro avec app.py)
# ═════════════════════════════════════════════════════════════════════════════

def get_app_data():
    """
    Récupère les données depuis current_app.config.
    Celles-ci sont maintenues à jour par _sync_app_config() dans app.py.
    """
    return (
        current_app.config.get('users',         []),
        current_app.config.get('tasks',         []),
        current_app.config.get('messages',      []),
        current_app.config.get('leaves',        []),
        current_app.config.get('activities',    []),
        current_app.config.get('feedbacks',     []),
        current_app.config.get('conversations', []),
    )


# ═════════════════════════════════════════════════════════════════════════════
# INITIALISATION DU MOTEUR ML
# ═════════════════════════════════════════════════════════════════════════════

@ml_bp.route('/initialize', methods=['POST'])
@require_admin
def initialize_ml_engine(user):
    """POST /api/ml/initialize — Initialise et entraîne tous les modèles ML."""
    try:
        users, tasks, messages, leaves, activities, feedbacks, conversations = get_app_data()
        result = orchestrator.initialize(
            users, tasks, messages, leaves, activities, feedbacks, conversations
        )
        return jsonify({
            'status':  'success',
            'message': 'Moteur ML initialisé avec succès',
            'details': result
        })
    except Exception as e:
        logger.error(f"Erreur initialisation ML: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# ANALYSE COMPLÈTE
# ═════════════════════════════════════════════════════════════════════════════

@ml_bp.route('/full-analysis', methods=['GET'])
@require_admin
def get_full_ml_analysis(user):
    """GET /api/ml/full-analysis — Analyse ML complète (turnover, sentiment, anomalies, réseau)."""
    try:
        users, tasks, messages, leaves, activities, feedbacks, conversations = get_app_data()
        result = orchestrator.run_full_analysis(
            users, tasks, messages, leaves, activities, feedbacks, conversations
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Erreur analyse complète: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# PRÉDICTION TURNOVER
# ═════════════════════════════════════════════════════════════════════════════

@ml_bp.route('/turnover/predictions', methods=['GET'])
@require_admin
def get_turnover_predictions(user):
    """
    GET /api/ml/turnover/predictions
    Query params: risk_level, department, limit
    """
    try:
        users, tasks, messages, leaves, activities, feedbacks, conversations = get_app_data()
        X, emp_ids = orchestrator.feature_engine.build_feature_matrix(
            users, tasks, messages, leaves, activities, feedbacks
        )

        if X.empty:
            return jsonify({'predictions': [], 'message': 'Données insuffisantes'}), 200

        if not orchestrator.turnover_model.is_trained:
            orchestrator.initialize(users, tasks, messages, leaves, activities, feedbacks, conversations)

        employees = {u['id']: u for u in users if u.get('role') == 'employee'}
        probs     = orchestrator.turnover_model.predict(X)
        predictions = []

        for i, (eid, prob) in enumerate(zip(emp_ids, probs)):
            emp        = employees.get(eid, {})
            risk_level = (
                'critical' if prob > 0.75 else
                'high'     if prob > 0.55 else
                'medium'   if prob > 0.35 else 'low'
            )

            if request.args.get('risk_level') and risk_level != request.args.get('risk_level'):
                continue
            if request.args.get('department') and emp.get('department') != request.args.get('department'):
                continue

            emp_X       = X.iloc[[i]]
            explanation = orchestrator.turnover_model.explain(emp_X, eid)
            actions     = orchestrator._get_retention_actions(explanation, risk_level)

            predictions.append({
                'employee': {
                    'id':         eid,
                    'full_name':  emp.get('full_name', ''),
                    'department': emp.get('department', ''),
                    'position':   emp.get('position', ''),
                },
                'ml_prediction': {
                    'turnover_probability': round(float(prob), 4),
                    'risk_level':           risk_level,
                    'confidence_interval': {
                        'lower': round(max(0, float(prob) - 0.08), 4),
                        'upper': round(min(1, float(prob) + 0.08), 4)
                    }
                },
                'explainability':      explanation,
                'recommended_actions': actions,
            })

        predictions.sort(key=lambda x: -x['ml_prediction']['turnover_probability'])
        limit = int(request.args.get('limit', 50))

        return jsonify({
            'predictions': predictions[:limit],
            'total':       len(predictions),
            'model_info': {
                'type':           'XGBoost + Calibration' if orchestrator.turnover_model.is_trained else 'Not trained',
                'explainability': 'SHAP TreeExplainer' if orchestrator.turnover_model.shap_explainer else 'Rule-based',
                'features_used':  orchestrator.turnover_model.feature_names
            },
            'risk_summary': {
                'critical': sum(1 for p in predictions if p['ml_prediction']['risk_level'] == 'critical'),
                'high':     sum(1 for p in predictions if p['ml_prediction']['risk_level'] == 'high'),
                'medium':   sum(1 for p in predictions if p['ml_prediction']['risk_level'] == 'medium'),
                'low':      sum(1 for p in predictions if p['ml_prediction']['risk_level'] == 'low'),
            }
        })

    except Exception as e:
        logger.error(f"Erreur prédiction turnover: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@ml_bp.route('/turnover/employee/<int:employee_id>', methods=['GET'])
@require_admin
def get_employee_turnover_detail(user, employee_id):
    """GET /api/ml/turnover/employee/{id} — Analyse détaillée d'un employé."""
    try:
        import pandas as pd
        users, tasks, messages, leaves, activities, feedbacks, conversations = get_app_data()

        emp = next((u for u in users if u['id'] == employee_id and u.get('role') == 'employee'), None)
        if not emp:
            return jsonify({'error': 'Employé introuvable'}), 404

        features = orchestrator.feature_engine.extract_employee_features(
            emp, tasks, messages, leaves, activities, feedbacks
        )
        X = pd.DataFrame([features])

        if not orchestrator.turnover_model.is_trained:
            orchestrator.initialize(users, tasks, messages, leaves, activities, feedbacks, conversations)

        prob        = float(orchestrator.turnover_model.predict(X)[0])
        explanation = orchestrator.turnover_model.explain(X, employee_id)

        dept_feedbacks = [f for f in feedbacks if f.get('department') == emp.get('department')]
        dept_sentiment = orchestrator.sentiment_analyzer.analyze_batch(dept_feedbacks[:20])

        return jsonify({
            'employee': {
                'id': emp['id'], 'full_name': emp.get('full_name', ''),
                'department': emp.get('department', ''), 'position': emp.get('position', '')
            },
            'turnover_risk': {
                'probability': round(prob, 4),
                'risk_level': ('critical' if prob > 0.75 else 'high' if prob > 0.55 else
                               'medium' if prob > 0.35 else 'low'),
            },
            'explainability':    explanation,
            'features':          {k: round(v, 3) for k, v in features.items()},
            'department_context': {
                'sentiment_summary': dept_sentiment.get('summary', {}),
                'dept_size': len([u for u in users if u.get('department') == emp.get('department')])
            },
            'retention_plan': orchestrator._get_retention_actions(explanation, 'high')
        })

    except Exception as e:
        logger.error(f"Erreur détail employé {employee_id}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# SENTIMENT
# ═════════════════════════════════════════════════════════════════════════════

@ml_bp.route('/sentiment/analyze', methods=['POST'])
@require_admin
def analyze_text_sentiment(user):
    """POST /api/ml/sentiment/analyze — Analyse de sentiment en temps réel."""
    try:
        data = request.get_json(silent=True) or {}
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'Texte requis'}), 400
        result = orchestrator.sentiment_analyzer.analyze_text(text)
        return jsonify({
            'input':      text[:200],
            'analysis':   result,
            'model_info': {
                'lexicon':    'VADER + HR domain lexicon',
                'classifier': 'TF-IDF + LinearSVC' if orchestrator.sentiment_analyzer.is_trained else 'VADER only'
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@ml_bp.route('/sentiment/batch-analysis', methods=['GET'])
@require_admin
def get_batch_sentiment_analysis(user):
    """GET /api/ml/sentiment/batch-analysis — Analyse NLP complète de tous les feedbacks."""
    try:
        _, _, _, _, _, feedbacks, _ = get_app_data()

        dept            = request.args.get('department')
        days            = int(request.args.get('days', 30))
        sentiment_filter= request.args.get('sentiment')
        cutoff          = datetime.now() - timedelta(days=days)

        filtered = [
            f for f in feedbacks
            if datetime.fromisoformat(f['created_at']) > cutoff
            and (not dept or f.get('department') == dept)
            and (not sentiment_filter or f.get('sentiment') == sentiment_filter)
        ]

        result = orchestrator.sentiment_analyzer.analyze_batch(filtered)

        dept_breakdown = {}
        for fb in result['feedbacks']:
            d     = fb.get('department', 'Unknown')
            label = fb.get('ml_analysis', {}).get('label', fb.get('sentiment', 'neutral'))
            if d not in dept_breakdown:
                dept_breakdown[d] = {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}
            dept_breakdown[d][label] = dept_breakdown[d].get(label, 0) + 1
            dept_breakdown[d]['total'] += 1

        dist         = result['summary'].get('sentiment_distribution', {})
        internal_nps = round(
            (dist.get('positive', 0) - dist.get('negative', 0)) * 100, 1
        )

        return jsonify({
            **result,
            'department_breakdown': dept_breakdown,
            'internal_nps_score':   internal_nps,
            'nps_interpretation': (
                'Excellent' if internal_nps > 50 else
                'Bon'       if internal_nps > 20 else
                'Moyen'     if internal_nps > 0  else 'Préoccupant'
            ),
            'filters_applied': {'department': dept, 'days': days, 'sentiment': sentiment_filter}
        })

    except Exception as e:
        logger.error(f"Erreur batch sentiment: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# ANOMALIES
# ═════════════════════════════════════════════════════════════════════════════

@ml_bp.route('/anomalies', methods=['GET'])
@require_admin
def get_anomalies(user):
    """GET /api/ml/anomalies — Détecte les anomalies comportementales."""
    try:
        users, tasks, messages, leaves, activities, feedbacks, _ = get_app_data()
        X, emp_ids = orchestrator.feature_engine.build_feature_matrix(
            users, tasks, messages, leaves, activities, feedbacks
        )
        if X.empty:
            return jsonify({'anomalies': [], 'total': 0})

        if not orchestrator.anomaly_detector.is_fitted:
            orchestrator.anomaly_detector.fit(X)

        anomalies        = orchestrator.anomaly_detector.detect(X, emp_ids)
        employees        = {u['id']: u for u in users if u.get('role') == 'employee'}
        severity_filter  = request.args.get('severity')
        type_filter      = request.args.get('type')

        enriched = []
        for a in anomalies:
            if severity_filter and a['max_severity'] != severity_filter:
                continue
            emp              = employees.get(a['employee_id'], {})
            filtered_anomalies = a['anomalies']
            if type_filter:
                filtered_anomalies = [an for an in filtered_anomalies if an.get('type') == type_filter]
                if not filtered_anomalies:
                    continue
            enriched.append({
                **a,
                'employee': {
                    'id': a['employee_id'], 'full_name': emp.get('full_name', ''),
                    'department': emp.get('department', ''), 'position': emp.get('position', '')
                },
                'anomalies': filtered_anomalies
            })

        return jsonify({
            'anomalies': enriched,
            'total': len(enriched),
            'severity_summary': {
                'high':   sum(1 for a in enriched if a['max_severity'] == 'high'),
                'medium': sum(1 for a in enriched if a['max_severity'] == 'medium')
            },
            'model_info': {
                'algorithm':     'Isolation Forest (sklearn)',
                'contamination': 0.05,
                'fitted':        orchestrator.anomaly_detector.is_fitted
            }
        })

    except Exception as e:
        logger.error(f"Erreur détection anomalies: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# FORECASTING
# ═════════════════════════════════════════════════════════════════════════════

@ml_bp.route('/forecast/<metric>', methods=['GET'])
@require_admin
def get_metric_forecast(user, metric):
    """GET /api/ml/forecast/{metric} — Prévisions temporelles (Prophet)."""
    SUPPORTED = ['productivity', 'sentiment', 'absenteeism', 'workload']
    if metric not in SUPPORTED:
        return jsonify({'error': f"Métrique inconnue. Supportées: {SUPPORTED}"}), 400

    try:
        users, tasks, messages, leaves, activities, feedbacks, _ = get_app_data()
        horizon = min(int(request.args.get('horizon', 30)), 180)
        dept    = request.args.get('department')
        time_series = []

        if metric == 'productivity':
            done_tasks  = [t for t in tasks if t['status'] == 'done']
            if dept:
                done_tasks = [t for t in done_tasks if t.get('department') == dept]
            time_series = orchestrator._build_daily_series(done_tasks, 'completed_at', 180)
        elif metric == 'sentiment':
            fbs = feedbacks if not dept else [f for f in feedbacks if f.get('department') == dept]
            time_series = orchestrator._build_sentiment_series(fbs, 180)
        elif metric == 'absenteeism':
            emp_leaves = leaves if not dept else [
                l for l in leaves if l.get('employee', {}).get('department') == dept
            ]
            time_series = orchestrator._build_daily_series(emp_leaves, 'created_at', 180)
        elif metric == 'workload':
            active = [t for t in tasks if t['status'] in ('todo', 'in_progress')]
            if dept:
                active = [t for t in active if t.get('department') == dept]
            time_series = orchestrator._build_daily_series(active, 'created_at', 180)

        if not time_series:
            return jsonify({'metric': metric, 'error': 'Données insuffisantes', 'data_points': 0}), 200

        result = orchestrator.forecaster.forecast(time_series, metric, horizon)
        return jsonify({
            **result,
            'filters':                {'department': dept, 'horizon': horizon},
            'historical_data_points': len(time_series),
        })

    except Exception as e:
        logger.error(f"Erreur forecast {metric}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# RÉSEAU DE COLLABORATION
# ═════════════════════════════════════════════════════════════════════════════

@ml_bp.route('/collaboration/network', methods=['GET'])
@require_admin
def get_collaboration_network(user):
    """GET /api/ml/collaboration/network — Analyse réseau (PageRank + silos)."""
    try:
        users, _, messages, _, _, _, conversations = get_app_data()
        result = orchestrator.collaboration_analyzer.analyze(users, messages, conversations)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Erreur réseau collaboration: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# TABLEAU DE BORD EXÉCUTIF
# ═════════════════════════════════════════════════════════════════════════════

@ml_bp.route('/executive-dashboard', methods=['GET'])
@require_admin
def get_executive_dashboard(user):
    """GET /api/ml/executive-dashboard — Vue synthétique KPIs ML."""
    try:
        users, tasks, messages, leaves, activities, feedbacks, conversations = get_app_data()
        full = orchestrator.run_full_analysis(
            users, tasks, messages, leaves, activities, feedbacks, conversations
        )

        health   = full.get('org_health_score', {})
        turnover = full.get('turnover_predictions', [])
        sentiment= full.get('sentiment_analysis', {})
        anomalies= full.get('anomalies', [])
        forecasts= full.get('forecasts', {})

        dashboard = {
            'generated_at': datetime.now().isoformat(),
            'org_health': {
                'global_score': health.get('global_score', 0),
                'status':       health.get('status', 'unknown'),
                'breakdown':    health.get('breakdown', {})
            },
            'turnover_kpis': {
                'employees_at_risk': sum(1 for r in turnover if r['risk_level'] in ('critical', 'high')),
                'critical_count':    sum(1 for r in turnover if r['risk_level'] == 'critical'),
                'avg_risk_score':    round(
                    sum(r['turnover_probability'] for r in turnover) / max(len(turnover), 1), 3
                ),
                'top_3_at_risk': [
                    {
                        'name':        r['employee_name'],
                        'department':  r['department'],
                        'probability': r['turnover_probability'],
                        'risk_level':  r['risk_level']
                    }
                    for r in turnover[:3]
                ]
            },
            'sentiment_kpis': {
                'nps_score': round(
                    (sentiment.get('summary', {}).get('sentiment_distribution', {}).get('positive', 0) -
                     sentiment.get('summary', {}).get('sentiment_distribution', {}).get('negative', 0)) * 100, 1
                ),
                'sentiment_distribution': sentiment.get('summary', {}).get('sentiment_distribution', {}),
                'top_topics': [
                    t[0] for t in sentiment.get('summary', {}).get('top_topics', [])[:3]
                ]
            },
            'critical_anomalies': [
                {
                    'employee_id': a['employee_id'],
                    'severity':    a['max_severity'],
                    'count':       a['anomaly_count'],
                    'main_issue':  a['anomalies'][0].get('description', '') if a['anomalies'] else ''
                }
                for a in anomalies if a['max_severity'] == 'high'
            ][:5],
            'forecast_summary': {
                metric: {
                    'trend':           data.get('trend', {}).get('direction', 'unknown'),
                    'next_month_avg':  data.get('next_month_summary', {}).get('predicted_avg')
                }
                for metric, data in forecasts.items()
                if isinstance(data, dict) and not data.get('error')
            },
            'priority_actions': full.get('executive_summary', {}).get('priority_actions', []),
            'alerts':           full.get('executive_summary', {}).get('key_alerts', [])
        }

        return jsonify(dashboard)

    except Exception as e:
        logger.error(f"Erreur dashboard exécutif: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# STATUT DES MODÈLES
# ═════════════════════════════════════════════════════════════════════════════

@ml_bp.route('/models/status', methods=['GET'])
@require_admin
def get_models_status(user):
    """GET /api/ml/models/status — Santé de tous les modèles ML."""
    try:
        from ml_engine import HAS_XGB, HAS_SKLEARN, HAS_SHAP, HAS_PROPHET, HAS_NLTK
    except ImportError:
        HAS_XGB = HAS_SKLEARN = HAS_SHAP = HAS_PROPHET = HAS_NLTK = False

    return jsonify({
        'models': {
            'turnover_predictor': {
                'name':       'XGBoost + Isotonic Calibration',
                'is_trained': orchestrator.turnover_model.is_trained,
                'has_shap':   orchestrator.turnover_model.shap_explainer is not None,
                'features':   orchestrator.turnover_model.feature_names,
                'n_features': len(orchestrator.turnover_model.feature_names)
            },
            'sentiment_analyzer': {
                'name':          'VADER + TF-IDF LinearSVC',
                'has_vader':     orchestrator.sentiment_analyzer.vader is not None,
                'is_ml_trained': orchestrator.sentiment_analyzer.is_trained,
            },
            'anomaly_detector': {
                'name':       'Isolation Forest',
                'is_fitted':  orchestrator.anomaly_detector.is_fitted,
                'contamination': 0.05
            },
            'forecaster': {
                'name':     'Facebook Prophet',
                'fallback': 'Linear regression + seasonal decomp'
            },
            'collaboration_analyzer': {
                'name':             'PageRank + Community detection',
                'always_available': True
            }
        },
        'libraries': {
            'xgboost':  HAS_XGB,
            'sklearn':  HAS_SKLEARN,
            'shap':     HAS_SHAP,
            'prophet':  HAS_PROPHET,
            'nltk':     HAS_NLTK,
        },
        'cache_status': {
            'cached_analyses': len(orchestrator._cache),
            'ttl_seconds':     orchestrator._cache_ttl
        },
        'checked_at': datetime.now().isoformat()
    })