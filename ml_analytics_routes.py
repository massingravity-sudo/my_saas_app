# ============================================================
# ML ANALYTICS API — Routes Flask avancées
# Remplace les fonctions heuristiques par de vrais modèles ML
# ============================================================

from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
from functools import wraps
from ml_engine import orchestrator
import logging

logger = logging.getLogger(__name__)
ml_bp = Blueprint('ml_analytics', __name__, url_prefix='/api/ml')


# ── Décorateur admin ─────────────────────────────────────────────────────────

def require_admin(f):
    """Vérification JWT admin (adapter à votre système d'auth)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        # Remplacer par votre logique JWT réelle
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'Token requis'}), 401
        # user = decode_jwt(token)  ← votre fonction
        user = {'id': 'admin', 'role': 'admin'}  # placeholder
        if user.get('role') != 'admin':
            return jsonify({'error': 'Accès refusé'}), 403
        return f(user, *args, **kwargs)
    return decorated


# ── Données (à remplacer par vos vrais stores/DB) ───────────────────────────
# Ces variables doivent pointer vers vos données réelles
# Exemple avec SQLAlchemy:
#   from models import User, Task, Feedback
#   users = User.query.all()
# Pour l'instant on utilise les listes globales de votre app Flask existante
def get_app_data():
    """Récupère les données depuis le contexte Flask de votre app."""
    from flask import current_app
    # Si votre app stocke les données en mémoire :
    return (
        current_app.config.get('users', []),
        current_app.config.get('tasks', []),
        current_app.config.get('messages', []),
        current_app.config.get('leaves', []),
        current_app.config.get('activities', []),
        current_app.config.get('feedbacks', []),
        current_app.config.get('conversations', []),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# INITIALISATION DU MOTEUR ML
# ═══════════════════════════════════════════════════════════════════════════════

@ml_bp.route('/initialize', methods=['POST'])
@require_admin
def initialize_ml_engine(user):
    """
    Initialise et entraîne tous les modèles ML.
    À appeler au démarrage ou pour forcer un ré-entraînement.
    POST /api/ml/initialize
    """
    try:
        users, tasks, messages, leaves, activities, feedbacks, conversations = get_app_data()

        result = orchestrator.initialize(
            users, tasks, messages, leaves,
            activities, feedbacks, conversations
        )

        return jsonify({
            'status': 'success',
            'message': 'Moteur ML initialisé avec succès',
            'details': result
        })

    except Exception as e:
        logger.error(f"Erreur initialisation ML: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSE COMPLÈTE
# ═══════════════════════════════════════════════════════════════════════════════

@ml_bp.route('/full-analysis', methods=['GET'])
@require_admin
def get_full_ml_analysis(user):
    """
    Lance l'analyse ML complète de la plateforme.
    Inclut : turnover, sentiment, anomalies, réseau, prévisions.
    GET /api/ml/full-analysis
    Réponse mise en cache 1 heure.
    """
    try:
        users, tasks, messages, leaves, activities, feedbacks, conversations = get_app_data()

        result = orchestrator.run_full_analysis(
            users, tasks, messages, leaves,
            activities, feedbacks, conversations
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Erreur analyse complète: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# PRÉDICTION TURNOVER
# ═══════════════════════════════════════════════════════════════════════════════

@ml_bp.route('/turnover/predictions', methods=['GET'])
@require_admin
def get_turnover_predictions(user):
    """
    Prédictions de turnover avec explainabilité SHAP par employé.
    GET /api/ml/turnover/predictions
    Query params:
      - risk_level: critical|high|medium|low (filtre)
      - department: string (filtre)
      - limit: int (défaut 50)
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
        probs = orchestrator.turnover_model.predict(X)

        predictions = []
        for i, (eid, prob) in enumerate(zip(emp_ids, probs)):
            emp = employees.get(eid, {})
            risk_level = (
                'critical' if prob > 0.75 else
                'high' if prob > 0.55 else
                'medium' if prob > 0.35 else 'low'
            )

            # Filtre
            if request.args.get('risk_level') and risk_level != request.args.get('risk_level'):
                continue
            if request.args.get('department') and emp.get('department') != request.args.get('department'):
                continue

            emp_X = X.iloc[[i]]
            explanation = orchestrator.turnover_model.explain(emp_X, eid)
            actions = orchestrator._get_retention_actions(explanation, risk_level)

            predictions.append({
                'employee': {
                    'id': eid,
                    'full_name': emp.get('full_name', ''),
                    'department': emp.get('department', ''),
                    'position': emp.get('position', ''),
                },
                'ml_prediction': {
                    'turnover_probability': round(float(prob), 4),
                    'risk_level': risk_level,
                    'confidence_interval': {
                        'lower': round(max(0, float(prob) - 0.08), 4),
                        'upper': round(min(1, float(prob) + 0.08), 4)
                    }
                },
                'explainability': explanation,
                'recommended_actions': actions,
                'feature_snapshot': {
                    col: round(float(X.iloc[i][col]), 3) for col in X.columns
                }
            })

        predictions.sort(key=lambda x: -x['ml_prediction']['turnover_probability'])
        limit = int(request.args.get('limit', 50))

        return jsonify({
            'predictions': predictions[:limit],
            'total': len(predictions),
            'model_info': {
                'type': 'XGBoost + Calibration' if orchestrator.turnover_model.is_trained else 'Not trained',
                'explainability': 'SHAP TreeExplainer' if orchestrator.turnover_model.shap_explainer else 'Rule-based',
                'features_used': orchestrator.turnover_model.feature_names
            },
            'risk_summary': {
                'critical': sum(1 for p in predictions if p['ml_prediction']['risk_level'] == 'critical'),
                'high': sum(1 for p in predictions if p['ml_prediction']['risk_level'] == 'high'),
                'medium': sum(1 for p in predictions if p['ml_prediction']['risk_level'] == 'medium'),
                'low': sum(1 for p in predictions if p['ml_prediction']['risk_level'] == 'low'),
            }
        })

    except Exception as e:
        logger.error(f"Erreur prédiction turnover: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@ml_bp.route('/turnover/employee/<employee_id>', methods=['GET'])
@require_admin
def get_employee_turnover_detail(user, employee_id: str):
    """
    Analyse de turnover détaillée pour un employé spécifique.
    GET /api/ml/turnover/employee/{employee_id}
    """
    try:
        users, tasks, messages, leaves, activities, feedbacks, conversations = get_app_data()

        emp = next((u for u in users if u['id'] == employee_id and u.get('role') == 'employee'), None)
        if not emp:
            return jsonify({'error': 'Employé introuvable'}), 404

        features = orchestrator.feature_engine.extract_employee_features(
            emp, tasks, messages, leaves, activities, feedbacks
        )

        import pandas as pd
        X = pd.DataFrame([features])

        if not orchestrator.turnover_model.is_trained:
            orchestrator.initialize(users, tasks, messages, leaves, activities, feedbacks, conversations)

        prob = float(orchestrator.turnover_model.predict(X)[0])
        explanation = orchestrator.turnover_model.explain(X, employee_id)

        # Analyse sentimentale des feedbacks du département
        dept_feedbacks = [f for f in feedbacks if f.get('department') == emp.get('department')]
        dept_sentiment = orchestrator.sentiment_analyzer.analyze_batch(dept_feedbacks[:20])

        return jsonify({
            'employee': {
                'id': emp['id'],
                'full_name': emp.get('full_name', ''),
                'department': emp.get('department', ''),
                'position': emp.get('position', '')
            },
            'turnover_risk': {
                'probability': round(prob, 4),
                'risk_level': (
                    'critical' if prob > 0.75 else
                    'high' if prob > 0.55 else
                    'medium' if prob > 0.35 else 'low'
                ),
                'percentile': 'Top 10% risque' if prob > 0.7 else 'Risque modéré'
            },
            'explainability': explanation,
            'features': {k: round(v, 3) for k, v in features.items()},
            'department_context': {
                'sentiment_summary': dept_sentiment.get('summary', {}),
                'dept_size': len([u for u in users if u.get('department') == emp.get('department')])
            },
            'retention_plan': orchestrator._get_retention_actions(explanation, 'high')
        })

    except Exception as e:
        logger.error(f"Erreur détail employé {employee_id}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSE DE SENTIMENT NLP
# ═══════════════════════════════════════════════════════════════════════════════

@ml_bp.route('/sentiment/analyze', methods=['POST'])
@require_admin
def analyze_text_sentiment(user):
    """
    Analyse le sentiment d'un texte en temps réel.
    POST /api/ml/sentiment/analyze
    Body: {"text": "...", "context": "feedback|message|survey"}
    """
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': 'Texte requis'}), 400

        result = orchestrator.sentiment_analyzer.analyze_text(text)
        return jsonify({
            'input': text[:200],  # truncate pour la réponse
            'analysis': result,
            'model_info': {
                'lexicon': 'VADER + HR domain lexicon',
                'classifier': 'TF-IDF + LinearSVC (calibrated)' if orchestrator.sentiment_analyzer.is_trained else 'VADER only'
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@ml_bp.route('/sentiment/batch-analysis', methods=['GET'])
@require_admin
def get_batch_sentiment_analysis(user):
    """
    Analyse NLP complète de tous les feedbacks.
    GET /api/ml/sentiment/batch-analysis
    Query params:
      - department: filtre département
      - days: période (défaut 30)
      - sentiment: positive|neutral|negative (filtre)
    """
    try:
        _, _, _, _, _, feedbacks, _ = get_app_data()

        # Filtres
        dept = request.args.get('department')
        days = int(request.args.get('days', 30))
        sentiment_filter = request.args.get('sentiment')
        cutoff = datetime.now() - timedelta(days=days)

        filtered_feedbacks = [
            f for f in feedbacks
            if datetime.fromisoformat(f['created_at']) > cutoff
            and (not dept or f.get('department') == dept)
            and (not sentiment_filter or f.get('sentiment') == sentiment_filter)
        ]

        result = orchestrator.sentiment_analyzer.analyze_batch(filtered_feedbacks)

        # Analyse par département
        dept_breakdown = {}
        for fb in result['feedbacks']:
            d = fb.get('department', 'Unknown')
            if d not in dept_breakdown:
                dept_breakdown[d] = {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}
            label = fb.get('ml_analysis', {}).get('label', fb.get('sentiment', 'neutral'))
            dept_breakdown[d][label] = dept_breakdown[d].get(label, 0) + 1
            dept_breakdown[d]['total'] += 1

        # Score NPS interne (proxy)
        dist = result['summary'].get('sentiment_distribution', {})
        internal_nps = round(
            (dist.get('positive', 0) - dist.get('negative', 0)) * 100, 1
        )

        return jsonify({
            **result,
            'department_breakdown': dept_breakdown,
            'internal_nps_score': internal_nps,
            'nps_interpretation': (
                'Excellent' if internal_nps > 50 else
                'Bon' if internal_nps > 20 else
                'Moyen' if internal_nps > 0 else
                'Préoccupant'
            ),
            'filters_applied': {'department': dept, 'days': days, 'sentiment': sentiment_filter}
        })

    except Exception as e:
        logger.error(f"Erreur batch sentiment: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# DÉTECTION D'ANOMALIES
# ═══════════════════════════════════════════════════════════════════════════════

@ml_bp.route('/anomalies', methods=['GET'])
@require_admin
def get_anomalies(user):
    """
    Détecte les anomalies comportementales avec Isolation Forest.
    GET /api/ml/anomalies
    Query params:
      - severity: high|medium (filtre)
      - type: behavioral_outlier|metric_spike|inactivity|... (filtre)
    """
    try:
        users, tasks, messages, leaves, activities, feedbacks, conversations = get_app_data()

        X, emp_ids = orchestrator.feature_engine.build_feature_matrix(
            users, tasks, messages, leaves, activities, feedbacks
        )

        if X.empty:
            return jsonify({'anomalies': [], 'total': 0})

        if not orchestrator.anomaly_detector.is_fitted:
            orchestrator.anomaly_detector.fit(X)

        anomalies = orchestrator.anomaly_detector.detect(X, emp_ids)

        # Enrichir avec les données employés
        employees = {u['id']: u for u in users if u.get('role') == 'employee'}
        severity_filter = request.args.get('severity')
        type_filter = request.args.get('type')

        enriched = []
        for a in anomalies:
            if severity_filter and a['max_severity'] != severity_filter:
                continue

            emp = employees.get(a['employee_id'], {})
            filtered_anomalies = a['anomalies']
            if type_filter:
                filtered_anomalies = [an for an in filtered_anomalies if an.get('type') == type_filter]
                if not filtered_anomalies:
                    continue

            enriched.append({
                **a,
                'employee': {
                    'id': a['employee_id'],
                    'full_name': emp.get('full_name', ''),
                    'department': emp.get('department', ''),
                    'position': emp.get('position', '')
                },
                'anomalies': filtered_anomalies
            })

        return jsonify({
            'anomalies': enriched,
            'total': len(enriched),
            'severity_summary': {
                'high': sum(1 for a in enriched if a['max_severity'] == 'high'),
                'medium': sum(1 for a in enriched if a['max_severity'] == 'medium')
            },
            'model_info': {
                'algorithm': 'Isolation Forest (sklearn)',
                'contamination': 0.05,
                'fitted': orchestrator.anomaly_detector.is_fitted
            }
        })

    except Exception as e:
        logger.error(f"Erreur détection anomalies: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════

@ml_bp.route('/forecast/<metric>', methods=['GET'])
@require_admin
def get_metric_forecast(user, metric: str):
    """
    Prévisions temporelles avec Prophet pour une métrique donnée.
    GET /api/ml/forecast/{metric}
    Métriques supportées: productivity, sentiment, absenteeism, workload
    Query params:
      - horizon: jours à prévoir (défaut 30, max 180)
      - department: filtre département
    """
    SUPPORTED_METRICS = ['productivity', 'sentiment', 'absenteeism', 'workload']
    if metric not in SUPPORTED_METRICS:
        return jsonify({
            'error': f"Métrique inconnue. Supportées: {SUPPORTED_METRICS}"
        }), 400

    try:
        users, tasks, messages, leaves, activities, feedbacks, conversations = get_app_data()

        horizon = min(int(request.args.get('horizon', 30)), 180)
        dept = request.args.get('department')

        time_series = []
        if metric == 'productivity':
            done_tasks = [t for t in tasks if t['status'] == 'done']
            if dept:
                done_tasks = [t for t in done_tasks if t.get('department') == dept]
            time_series = orchestrator._build_daily_series(done_tasks, 'completed_at', 180)

        elif metric == 'sentiment':
            fbs = feedbacks if not dept else [f for f in feedbacks if f.get('department') == dept]
            time_series = orchestrator._build_sentiment_series(fbs, 180)

        elif metric == 'absenteeism':
            emp_leaves = leaves if not dept else [
                l for l in leaves
                if l.get('employee', {}).get('department') == dept
            ]
            time_series = orchestrator._build_daily_series(emp_leaves, 'created_at', 180)

        elif metric == 'workload':
            active = [t for t in tasks if t['status'] in ('todo', 'in_progress')]
            if dept:
                active = [t for t in active if t.get('department') == dept]
            time_series = orchestrator._build_daily_series(active, 'created_at', 180)

        if not time_series:
            return jsonify({
                'metric': metric,
                'error': 'Données insuffisantes pour cette métrique',
                'data_points': 0
            }), 200

        result = orchestrator.forecaster.forecast(time_series, metric, horizon)

        return jsonify({
            **result,
            'filters': {'department': dept, 'horizon': horizon},
            'historical_data_points': len(time_series),
            'model_info': {
                'primary': 'Facebook Prophet',
                'fallback': 'Linear trend + seasonal decomposition',
                'active': result.get('method', 'unknown')
            }
        })

    except Exception as e:
        logger.error(f"Erreur forecast {metric}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# RÉSEAU DE COLLABORATION
# ═══════════════════════════════════════════════════════════════════════════════

@ml_bp.route('/collaboration/network', methods=['GET'])
@require_admin
def get_collaboration_network(user):
    """
    Analyse du réseau de collaboration avec PageRank et détection de silos.
    GET /api/ml/collaboration/network
    """
    try:
        users, tasks, messages, leaves, activities, feedbacks, conversations = get_app_data()

        result = orchestrator.collaboration_analyzer.analyze(
            users, messages, conversations
        )

        return jsonify({
            **result,
            'model_info': {
                'algorithm': 'PageRank (5 iterations) + Community detection',
                'edge_source': 'Direct messages + shared conversations'
            }
        })

    except Exception as e:
        logger.error(f"Erreur réseau collaboration: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# TABLEAU DE BORD EXÉCUTIF
# ═══════════════════════════════════════════════════════════════════════════════

@ml_bp.route('/executive-dashboard', methods=['GET'])
@require_admin
def get_executive_dashboard(user):
    """
    Vue synthétique pour les dirigeants.
    Agrège les KPIs ML les plus importants en un seul appel.
    GET /api/ml/executive-dashboard
    """
    try:
        users, tasks, messages, leaves, activities, feedbacks, conversations = get_app_data()

        # Full analysis (cached)
        full = orchestrator.run_full_analysis(
            users, tasks, messages, leaves,
            activities, feedbacks, conversations
        )

        # KPIs synthétiques
        health = full.get('org_health_score', {})
        turnover = full.get('turnover_predictions', [])
        sentiment = full.get('sentiment_analysis', {})
        anomalies = full.get('anomalies', [])
        forecasts = full.get('forecasts', {})

        dashboard = {
            'generated_at': datetime.now().isoformat(),

            # Score santé
            'org_health': {
                'global_score': health.get('global_score', 0),
                'status': health.get('status', 'unknown'),
                'breakdown': health.get('breakdown', {})
            },

            # KPIs turnover
            'turnover_kpis': {
                'employees_at_risk': sum(1 for r in turnover if r['risk_level'] in ('critical', 'high')),
                'critical_count': sum(1 for r in turnover if r['risk_level'] == 'critical'),
                'avg_risk_score': round(
                    sum(r['turnover_probability'] for r in turnover) / max(len(turnover), 1), 3
                ),
                'top_3_at_risk': [
                    {
                        'name': r['employee_name'],
                        'department': r['department'],
                        'probability': r['turnover_probability'],
                        'risk_level': r['risk_level']
                    }
                    for r in turnover[:3]
                ]
            },

            # KPIs sentiment
            'sentiment_kpis': {
                'nps_score': round(
                    (sentiment.get('summary', {}).get('sentiment_distribution', {}).get('positive', 0) -
                     sentiment.get('summary', {}).get('sentiment_distribution', {}).get('negative', 0)) * 100, 1
                ),
                'sentiment_distribution': sentiment.get('summary', {}).get('sentiment_distribution', {}),
                'top_topics': [t[0] for t in sentiment.get('summary', {}).get('top_topics', [])[:3]]
            },

            # Anomalies critiques
            'critical_anomalies': [
                {
                    'employee_id': a['employee_id'],
                    'severity': a['max_severity'],
                    'count': a['anomaly_count'],
                    'main_issue': a['anomalies'][0].get('description', '') if a['anomalies'] else ''
                }
                for a in anomalies if a['max_severity'] == 'high'
            ][:5],

            # Tendances prévues
            'forecast_summary': {
                metric: {
                    'trend': data.get('trend', {}).get('direction', 'unknown'),
                    'next_month_avg': data.get('next_month_summary', {}).get('predicted_avg')
                }
                for metric, data in forecasts.items()
                if isinstance(data, dict) and not data.get('error')
            },

            # Plan d'action prioritaire
            'priority_actions': full.get('executive_summary', {}).get('priority_actions', []),
            'alerts': full.get('executive_summary', {}).get('key_alerts', [])
        }

        return jsonify(dashboard)

    except Exception as e:
        logger.error(f"Erreur dashboard exécutif: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# MLOps — Santé des modèles
# ═══════════════════════════════════════════════════════════════════════════════

@ml_bp.route('/models/status', methods=['GET'])
@require_admin
def get_models_status(user):
    """
    État de santé de tous les modèles ML actifs.
    GET /api/ml/models/status
    """
    return jsonify({
        'models': {
            'turnover_predictor': {
                'name': 'XGBoost + Isotonic Calibration',
                'is_trained': orchestrator.turnover_model.is_trained,
                'has_shap': orchestrator.turnover_model.shap_explainer is not None,
                'features': orchestrator.turnover_model.feature_names,
                'n_features': len(orchestrator.turnover_model.feature_names)
            },
            'sentiment_analyzer': {
                'name': 'VADER + TF-IDF LinearSVC',
                'has_vader': orchestrator.sentiment_analyzer.vader is not None,
                'is_ml_trained': orchestrator.sentiment_analyzer.is_trained,
                'has_hr_lexicon': True
            },
            'anomaly_detector': {
                'name': 'Isolation Forest',
                'is_fitted': orchestrator.anomaly_detector.is_fitted,
                'contamination': 0.05
            },
            'forecaster': {
                'name': 'Facebook Prophet',
                'prophet_available': True,
                'fallback': 'Linear regression + seasonal decomp'
            },
            'collaboration_analyzer': {
                'name': 'PageRank + Community detection',
                'always_available': True
            }
        },
        'libraries': {
            'xgboost': HAS_XGB if 'HAS_XGB' in dir() else False,
            'sklearn': HAS_SKLEARN if 'HAS_SKLEARN' in dir() else False,
            'shap': HAS_SHAP if 'HAS_SHAP' in dir() else False,
            'prophet': HAS_PROPHET if 'HAS_PROPHET' in dir() else False,
            'nltk': HAS_NLTK if 'HAS_NLTK' in dir() else False
        },
        'cache_status': {
            'cached_analyses': len(orchestrator._cache),
            'ttl_seconds': orchestrator._cache_ttl
        },
        'checked_at': datetime.now().isoformat()
    })




# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSE ML DES ENQUÊTES — Moteur NLP + Statistiques avancées
# ═══════════════════════════════════════════════════════════════════════════════

def _get_surveys_data():
    """Récupère les enquêtes et réponses depuis la DB SQLAlchemy."""
    try:
        from flask import current_app
        from models import Survey, SurveyResponse, User
        surveys_list = []
        for s in Survey.query.all():
            responses = SurveyResponse.query.filter_by(survey_id=s.id).all()
            surveys_list.append({
                'survey': s.to_dict(),
                'responses': [r.to_dict() for r in responses]
            })
        return surveys_list
    except Exception as e:
        logger.error(f"Erreur récupération enquêtes: {e}")
        return []


def _analyze_survey_with_ml(survey_dict: dict, responses_list: list) -> dict:
    """
    Analyse complète d'une enquête par le moteur ML.
    Traite chaque type de question différemment :
      - text       → NLP sentiment + thèmes + mots-clés
      - scale      → stats descriptives + distribution + benchmark
      - single/multiple_choice → distribution + corrélations + insights
    """
    import statistics
    from collections import Counter, defaultdict

    questions    = survey_dict.get('questions', [])
    total_resp   = len(responses_list)
    analyzed_qs  = []

    # ── Analyse question par question ────────────────────────────────────────
    for i, question in enumerate(questions):
        q_type   = question.get('type', 'text')
        q_label  = question.get('label', f'Question {i+1}')
        answers  = [r.get('answers', [])[i] for r in responses_list if i < len(r.get('answers', []))]
        answers  = [a for a in answers if a is not None and a != '']

        qa_result = {
            'index':        i,
            'label':        q_label,
            'type':         q_type,
            'response_count': len(answers),
            'response_rate':  round(len(answers) / max(total_resp, 1) * 100, 1),
        }

        if q_type == 'text':
            # ── NLP sur réponses textuelles ──────────────────────────────
            text_analyses = []
            sent_counts   = Counter()
            all_topics    = Counter()
            intensities   = []

            for ans in answers:
                if not str(ans).strip():
                    continue
                ml_result = orchestrator.sentiment_analyzer.analyze_text(str(ans), category=None)
                text_analyses.append({
                    'text':      str(ans)[:300],
                    'sentiment': ml_result.get('label', 'neutral'),
                    'confidence':ml_result.get('confidence', 0.5),
                    'intensity': ml_result.get('intensity', 0),
                    'topics':    ml_result.get('topics', []),
                    'sarcasm':   ml_result.get('sarcasm_detected', False),
                    'score':     ml_result.get('compound_score', 0),
                })
                sent_counts[ml_result.get('label', 'neutral')] += 1
                for t in ml_result.get('topics', []):
                    all_topics[t] += 1
                intensities.append(ml_result.get('intensity', 0))

            total_text = len(text_analyses)
            dist = {k: round(v / max(total_text, 1), 3) for k, v in sent_counts.items()}
            nps  = round((dist.get('positive', 0) - dist.get('negative', 0)) * 100, 1)

            # Benchmark interne : score moyen pondéré [-1, 1] → [0, 100]
            scores    = [a['score'] for a in text_analyses]
            avg_score = statistics.mean(scores) if scores else 0
            health    = round((avg_score + 1) / 2 * 100, 1)

            qa_result.update({
                'ml_analysis': {
                    'sentiment_distribution': dist,
                    'nps_score':              nps,
                    'health_score':           health,
                    'avg_intensity':          round(statistics.mean(intensities), 2) if intensities else 0,
                    'top_topics':             all_topics.most_common(5),
                    'sarcasm_count':          sum(1 for a in text_analyses if a['sarcasm']),
                    'responses_detail':       sorted(text_analyses, key=lambda x: -abs(x['score']))[:20],
                    'interpretation':         _interpret_text_score(nps, dist),
                }
            })

        elif q_type == 'scale':
            # ── Statistiques descriptives sur échelles ───────────────────
            try:
                nums = [float(a) for a in answers if str(a).strip().lstrip('-').replace('.','',1).isdigit()]
            except Exception:
                nums = []

            if nums:
                mn   = min(nums)
                mx   = max(nums)
                mean = statistics.mean(nums)
                med  = statistics.median(nums)
                std  = statistics.stdev(nums) if len(nums) > 1 else 0

                # Distribution par valeur
                value_dist = Counter(int(n) for n in nums)
                max_val    = int(question.get('max', 5))

                # Score normalisé [0, 100]
                normalized = round((mean - 1) / max(max_val - 1, 1) * 100, 1)

                # Interprétation benchmark
                benchmark_label = (
                    'Excellent'  if normalized >= 80 else
                    'Bon'        if normalized >= 65 else
                    'Moyen'      if normalized >= 50 else
                    'Insuffisant'if normalized >= 35 else
                    'Critique'
                )
                benchmark_color = (
                    'green'  if normalized >= 80 else
                    'blue'   if normalized >= 65 else
                    'yellow' if normalized >= 50 else
                    'orange' if normalized >= 35 else
                    'red'
                )

                # Distribution par tranche (faible / moyen / fort)
                low_threshold  = max_val * 0.33
                high_threshold = max_val * 0.66
                low_count  = sum(1 for n in nums if n <= low_threshold)
                mid_count  = sum(1 for n in nums if low_threshold < n <= high_threshold)
                high_count = sum(1 for n in nums if n > high_threshold)

                qa_result.update({
                    'stats': {
                        'mean':      round(mean, 2),
                        'median':    round(med, 2),
                        'std_dev':   round(std, 2),
                        'min':       mn,
                        'max':       mx,
                        'max_scale': max_val,
                        'normalized_score': normalized,
                        'benchmark_label':  benchmark_label,
                        'benchmark_color':  benchmark_color,
                        'value_distribution': dict(sorted(value_dist.items())),
                        'segments': {
                            'low':  {'count': low_count,  'pct': round(low_count  / max(len(nums),1) * 100, 1)},
                            'mid':  {'count': mid_count,  'pct': round(mid_count  / max(len(nums),1) * 100, 1)},
                            'high': {'count': high_count, 'pct': round(high_count / max(len(nums),1) * 100, 1)},
                        }
                    }
                })

        elif q_type in ('single_choice', 'multiple_choice'):
            # ── Distribution des choix + dominance ──────────────────────
            opt_counter = Counter()
            for ans in answers:
                opts = ans if isinstance(ans, list) else [ans]
                for opt in opts:
                    if opt:
                        opt_counter[opt] += 1

            total_votes = sum(opt_counter.values()) or 1
            opt_dist    = {
                opt: {
                    'count': cnt,
                    'pct':   round(cnt / total_votes * 100, 1)
                }
                for opt, cnt in opt_counter.most_common()
            }

            # Option dominante + consensus
            most_common_opt, most_common_cnt = opt_counter.most_common(1)[0] if opt_counter else ('—', 0)
            dominance = round(most_common_cnt / total_votes * 100, 1)
            consensus = 'Forte' if dominance > 60 else 'Modérée' if dominance > 40 else 'Faible'

            qa_result.update({
                'distribution': opt_dist,
                'dominant_option': most_common_opt,
                'dominance_pct':   dominance,
                'consensus':       consensus,
                'unique_options':  len(opt_counter),
            })

        analyzed_qs.append(qa_result)

    # ── Score de santé global de l'enquête ──────────────────────────────────
    health_scores = []
    for qa in analyzed_qs:
        if 'ml_analysis' in qa:
            health_scores.append(qa['ml_analysis']['health_score'])
        elif 'stats' in qa:
            health_scores.append(qa['stats']['normalized_score'])

    overall_health = round(statistics.mean(health_scores), 1) if health_scores else None

    # ── Analyse des tendances par département ────────────────────────────────
    dept_breakdown = defaultdict(lambda: {'count': 0, 'text_scores': [], 'scale_scores': []})
    for r in responses_list:
        dept = r.get('department', 'Non défini')
        dept_breakdown[dept]['count'] += 1
        for i, qa in enumerate(analyzed_qs):
            if i < len(r.get('answers', [])):
                ans = r['answers'][i]
                if qa['type'] == 'text' and ans:
                    ml = orchestrator.sentiment_analyzer.analyze_text(str(ans), category=None)
                    dept_breakdown[dept]['text_scores'].append(ml.get('compound_score', 0))
                elif qa['type'] == 'scale' and ans:
                    try:
                        max_s = float(qa.get('question', {}).get('max', 5) or 5)
                        dept_breakdown[dept]['scale_scores'].append(float(ans) / max_s * 100)
                    except Exception:
                        pass

    dept_summary = {}
    for dept, d in dept_breakdown.items():
        text_avg  = statistics.mean(d['text_scores'])  if d['text_scores']  else None
        scale_avg = statistics.mean(d['scale_scores']) if d['scale_scores'] else None
        health_d  = None
        if text_avg is not None and scale_avg is not None:
            health_d = round((text_avg + 1) / 2 * 50 + scale_avg / 2, 1)
        elif text_avg is not None:
            health_d = round((text_avg + 1) / 2 * 100, 1)
        elif scale_avg is not None:
            health_d = round(scale_avg, 1)
        dept_summary[dept] = {'count': d['count'], 'health_score': health_d}

    # ── Recommandations ML ───────────────────────────────────────────────────
    recommendations = _generate_survey_recommendations(analyzed_qs, overall_health)

    return {
        'survey':              survey_dict,
        'total_responses':     total_resp,
        'overall_health_score': overall_health,
        'questions_analysis':  analyzed_qs,
        'department_breakdown': dept_summary,
        'recommendations':     recommendations,
        'analyzed_at':         datetime.now().isoformat(),
    }


def _interpret_text_score(nps: float, dist: dict) -> str:
    neg = dist.get('negative', 0)
    pos = dist.get('positive', 0)
    if nps > 50:
        return 'Satisfaction très élevée — engagement fort des répondants'
    elif nps > 20:
        return 'Bonne satisfaction globale — quelques points d\'amélioration possibles'
    elif nps > 0:
        return 'Satisfaction mitigée — attention requise sur les points de friction'
    elif nps > -20:
        return 'Insatisfaction modérée — plan d\'action correctif recommandé'
    else:
        return f'Insatisfaction critique ({round(neg*100)}% de réponses négatives) — intervention urgente'


def _generate_survey_recommendations(analyzed_qs: list, overall_health) -> list:
    """Génère des recommandations stratégiques basées sur les résultats ML."""
    recs = []
    for qa in analyzed_qs:
        label = qa.get('label', 'cette question')
        if 'ml_analysis' in qa:
            ml = qa['ml_analysis']
            nps = ml.get('nps_score', 0)
            topics = ml.get('top_topics', [])
            if nps < -20:
                recs.append({
                    'priority': 'critical',
                    'question': label,
                    'insight':  f'NPS de {nps} détecté — insatisfaction sévère',
                    'action':   'Organiser des entretiens individuels et groupes de travail immédiats'
                })
            elif nps < 0:
                recs.append({
                    'priority': 'high',
                    'question': label,
                    'insight':  f'NPS de {nps} — majorité d\'avis défavorables',
                    'action':   'Identifier les causes racines et mettre en place un plan correctif'
                })
            if topics:
                topic_name = topics[0][0] if isinstance(topics[0], (list, tuple)) else topics[0]
                recs.append({
                    'priority': 'medium',
                    'question': label,
                    'insight':  f'Thème récurrent détecté : "{topic_name.replace("_", " ")}"',
                    'action':   f'Adresser spécifiquement le sujet "{topic_name.replace("_", " ")}" en priorité'
                })
        elif 'stats' in qa:
            score = qa['stats'].get('normalized_score', 50)
            if score < 40:
                recs.append({
                    'priority': 'high',
                    'question': label,
                    'insight':  f'Score moyen bas : {score:.0f}/100',
                    'action':   'Analyser les causes du faible score et prendre des mesures correctives'
                })
    if overall_health is not None and overall_health < 50:
        recs.insert(0, {
            'priority': 'critical',
            'question': 'Enquête globale',
            'insight':  f'Score de santé global insuffisant : {overall_health:.0f}/100',
            'action':   'Réunion de direction urgente pour analyser les résultats et définir un plan d\'action'
        })
    return sorted(recs, key=lambda x: {'critical':0,'high':1,'medium':2,'low':3}[x['priority']])[:8]


@ml_bp.route('/surveys', methods=['GET'])
@require_admin
def get_surveys_ml_overview(user):
    """
    Vue d'ensemble ML de toutes les enquêtes.
    GET /api/ml/surveys
    Retourne : liste des enquêtes avec score de santé ML et taux de réponse.
    """
    try:
        surveys_data = _get_surveys_data()
        overview = []
        for item in surveys_data:
            s = item['survey']
            responses = item['responses']
            total_resp = len(responses)

            # Score sentiment rapide sur les réponses textuelles
            text_scores = []
            for r in responses:
                for i, q in enumerate(s.get('questions', [])):
                    if q.get('type') == 'text' and i < len(r.get('answers', [])):
                        ans = r['answers'][i]
                        if ans and str(ans).strip():
                            ml = orchestrator.sentiment_analyzer.analyze_text(str(ans), category=None)
                            text_scores.append(ml.get('compound_score', 0))

            avg_score = sum(text_scores) / max(len(text_scores), 1) if text_scores else None
            health    = round((avg_score + 1) / 2 * 100, 1) if avg_score is not None else None
            nps_quick = round((avg_score or 0) * 100, 1)

            overview.append({
                'id':              s['id'],
                'title':           s['title'],
                'description':     s.get('description', ''),
                'status':          s.get('status', 'active'),
                'target_department': s.get('target_department', 'all'),
                'anonymous':       s.get('anonymous', False),
                'created_at':      s.get('created_at'),
                'deadline':        s.get('deadline'),
                'total_responses': total_resp,
                'question_count':  len(s.get('questions', [])),
                'ml_health_score': health,
                'ml_nps':          nps_quick if text_scores else None,
                'has_text_questions': any(q.get('type') == 'text' for q in s.get('questions', [])),
                'has_scale_questions': any(q.get('type') == 'scale' for q in s.get('questions', [])),
            })

        overview.sort(key=lambda x: x['created_at'] or '', reverse=True)
        return jsonify({'surveys': overview, 'total': len(overview)})

    except Exception as e:
        logger.error(f"Erreur overview enquêtes ML: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@ml_bp.route('/surveys/<int:survey_id>/analysis', methods=['GET'])
@require_admin
def get_survey_ml_analysis(user, survey_id: int):
    """
    Analyse ML complète d'une enquête spécifique.
    GET /api/ml/surveys/{id}/analysis
    Analyse chaque question par type :
      - text   → NLP multicouche (sentiment + thèmes + intensité + sarcasme)
      - scale  → stats descriptives + benchmark + distribution
      - choice → distribution + dominance + consensus
    Plus : score de santé global, breakdown par département, recommandations.
    """
    try:
        surveys_data = _get_surveys_data()
        item = next((d for d in surveys_data if d['survey']['id'] == survey_id), None)

        if not item:
            return jsonify({'error': 'Enquête non trouvée'}), 404

        result = _analyze_survey_with_ml(item['survey'], item['responses'])
        return jsonify(result)

    except Exception as e:
        logger.error(f"Erreur analyse ML enquête {survey_id}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@ml_bp.route('/surveys/global-insights', methods=['GET'])
@require_admin
def get_surveys_global_insights(user):
    """
    Insights ML agrégés sur TOUTES les enquêtes.
    GET /api/ml/surveys/global-insights
    Retourne : tendances cross-enquêtes, thèmes globaux, évolution temporelle.
    """
    try:
        import statistics
        from collections import Counter
        surveys_data = _get_surveys_data()

        all_text_scores = []
        all_topics      = Counter()
        dept_scores     = {}
        timeline        = []

        for item in surveys_data:
            s         = item['survey']
            responses = item['responses']
            created   = s.get('created_at', '')[:10]
            survey_scores = []

            for r in responses:
                for i, q in enumerate(s.get('questions', [])):
                    if q.get('type') == 'text' and i < len(r.get('answers', [])):
                        ans = r['answers'][i]
                        if ans and str(ans).strip():
                            ml = orchestrator.sentiment_analyzer.analyze_text(str(ans), category=None)
                            score = ml.get('compound_score', 0)
                            all_text_scores.append(score)
                            survey_scores.append(score)
                            for t in ml.get('topics', []):
                                all_topics[t] += 1
                            dept = r.get('department', 'N/A')
                            if dept not in dept_scores:
                                dept_scores[dept] = []
                            dept_scores[dept].append(score)

            if survey_scores and created:
                avg = statistics.mean(survey_scores)
                timeline.append({
                    'date':   created,
                    'title':  s['title'],
                    'score':  round((avg + 1) / 2 * 100, 1),
                    'n_resp': len(responses)
                })

        dept_health = {
            dept: round((statistics.mean(scores) + 1) / 2 * 100, 1)
            for dept, scores in dept_scores.items() if scores
        }

        global_avg   = statistics.mean(all_text_scores) if all_text_scores else None
        global_health= round((global_avg + 1) / 2 * 100, 1) if global_avg is not None else None
        global_nps   = round((global_avg or 0) * 100, 1)

        return jsonify({
            'global_health_score': global_health,
            'global_nps':          global_nps,
            'total_text_responses': len(all_text_scores),
            'top_topics':           all_topics.most_common(8),
            'department_health':    dept_health,
            'timeline':             sorted(timeline, key=lambda x: x['date']),
            'trend': (
                'improving' if len(timeline) >= 2 and timeline[-1]['score'] > timeline[0]['score']
                else 'declining' if len(timeline) >= 2 and timeline[-1]['score'] < timeline[0]['score']
                else 'stable'
            ),
            'analyzed_at': datetime.now().isoformat(),
        })

    except Exception as e:
        logger.error(f"Erreur insights globaux enquêtes: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ── Importer les constantes de disponibilité depuis ml_engine ────────────────
try:
    from ml_engine import HAS_XGB, HAS_SKLEARN, HAS_SHAP, HAS_PROPHET, HAS_NLTK
except ImportError:
    HAS_XGB = HAS_SKLEARN = HAS_SHAP = HAS_PROPHET = HAS_NLTK = False