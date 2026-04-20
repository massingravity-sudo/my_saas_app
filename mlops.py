# ============================================================
# MLOps LAYER — Gestion du cycle de vie des modèles ML
# Versioning · Drift detection · Auto-retrain · A/B testing
# ============================================================

import json
import hashlib
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / 'ml_models'
MODELS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MODEL REGISTRY — Versioning des modèles
# ═══════════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    """
    Registre centralisé des modèles ML.
    Gère : versions, métriques, déploiement, rollback.
    Inspiré de MLflow (version simplifiée sans dépendance externe).
    """

    REGISTRY_FILE = MODELS_DIR / 'registry.json'

    def __init__(self):
        self.registry: dict = self._load_registry()

    def _load_registry(self) -> dict:
        if self.REGISTRY_FILE.exists():
            with open(self.REGISTRY_FILE) as f:
                return json.load(f)
        return {'models': {}, 'experiments': []}

    def _save_registry(self):
        with open(self.REGISTRY_FILE, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)

    def register_model(
        self, model_name: str, version: str,
        metrics: dict, params: dict,
        model_path: str, status: str = 'staging'
    ) -> dict:
        """
        Enregistre une nouvelle version d'un modèle.
        Status: staging → production → archived
        """
        if model_name not in self.registry['models']:
            self.registry['models'][model_name] = {'versions': [], 'production': None}

        entry = {
            'version': version,
            'registered_at': datetime.now().isoformat(),
            'metrics': metrics,
            'params': params,
            'model_path': model_path,
            'status': status,
            'model_hash': self._hash_model_path(model_path)
        }

        self.registry['models'][model_name]['versions'].append(entry)
        self._save_registry()

        logger.info(f"Modèle enregistré: {model_name} v{version} ({status})")
        return entry

    def promote_to_production(self, model_name: str, version: str) -> bool:
        """Promeut une version en production (l'ancienne passe en archived)."""
        if model_name not in self.registry['models']:
            return False

        model_data = self.registry['models'][model_name]

        # Archiver l'ancienne version de production
        if model_data['production']:
            for v in model_data['versions']:
                if v['version'] == model_data['production']:
                    v['status'] = 'archived'

        # Promouvoir la nouvelle
        for v in model_data['versions']:
            if v['version'] == version:
                v['status'] = 'production'
                model_data['production'] = version
                self._save_registry()
                logger.info(f"Modèle promu en production: {model_name} v{version}")
                return True

        return False

    def get_production_version(self, model_name: str) -> Optional[dict]:
        """Retourne la version de production actuelle."""
        if model_name not in self.registry['models']:
            return None
        model_data = self.registry['models'][model_name]
        prod_version = model_data.get('production')
        if prod_version:
            return next(
                (v for v in model_data['versions'] if v['version'] == prod_version),
                None
            )
        return None

    def get_model_history(self, model_name: str) -> list:
        """Historique complet des versions d'un modèle."""
        if model_name not in self.registry['models']:
            return []
        return sorted(
            self.registry['models'][model_name]['versions'],
            key=lambda x: x['registered_at'],
            reverse=True
        )

    def rollback(self, model_name: str) -> Optional[dict]:
        """Rollback vers la version précédente de production."""
        history = self.get_model_history(model_name)
        archived = [v for v in history if v['status'] == 'archived']
        if not archived:
            return None
        latest_archived = archived[0]
        self.promote_to_production(model_name, latest_archived['version'])
        return latest_archived

    def log_experiment(self, experiment_name: str, params: dict, metrics: dict) -> str:
        """Enregistre un experiment de training."""
        exp_id = hashlib.md5(
            f"{experiment_name}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]

        self.registry['experiments'].append({
            'id': exp_id,
            'name': experiment_name,
            'params': params,
            'metrics': metrics,
            'run_at': datetime.now().isoformat()
        })
        self._save_registry()
        return exp_id

    def _hash_model_path(self, path: str) -> str:
        p = Path(path)
        if p.exists():
            with open(p, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()[:12]
        return 'file_not_found'

    def get_summary(self) -> dict:
        return {
            'total_models': len(self.registry['models']),
            'total_experiments': len(self.registry['experiments']),
            'models': {
                name: {
                    'versions': len(data['versions']),
                    'production_version': data.get('production'),
                    'latest_metrics': next(
                        (v['metrics'] for v in data['versions'] if v['status'] == 'production'),
                        None
                    )
                }
                for name, data in self.registry['models'].items()
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DRIFT DETECTOR — Détection de dérive des données et des modèles
# ═══════════════════════════════════════════════════════════════════════════════

class DriftDetector:
    """
    Détecte la dérive des données (data drift) et la dérive du modèle (model drift).
    Algorithmes : PSI (Population Stability Index), KS-test, Chi-squared.
    Déclenche un ré-entraînement si la dérive dépasse les seuils.
    """

    PSI_THRESHOLD = 0.2       # >0.2 = dérive significative
    KS_PVALUE_THRESHOLD = 0.05  # p < 0.05 = distributions différentes

    def __init__(self):
        self.reference_distributions: dict = {}
        self.drift_history: list = []

    def set_reference(self, X: pd.DataFrame, name: str = 'training'):
        """Enregistre la distribution de référence (données d'entraînement)."""
        self.reference_distributions[name] = {
            col: {
                'mean': float(X[col].mean()),
                'std': float(X[col].std()) or 1.0,
                'quantiles': X[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
                'histogram': np.histogram(X[col].dropna(), bins=10)
            }
            for col in X.columns
        }
        logger.info(f"Distribution de référence '{name}' enregistrée ({len(X)} samples)")

    def compute_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Population Stability Index (PSI).
        PSI < 0.1: stable, 0.1-0.2: minor drift, > 0.2: major drift.
        """
        eps = 1e-10

        # Créer les bins sur la distribution attendue
        min_val, max_val = min(expected.min(), actual.min()), max(expected.max(), actual.max())
        breakpoints = np.linspace(min_val, max_val, bins + 1)

        expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected) + eps
        actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual) + eps

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    def detect_drift(self, X_current: pd.DataFrame, reference_name: str = 'training') -> dict:
        """
        Analyse la dérive des données actuelles vs la référence.
        """
        if reference_name not in self.reference_distributions:
            return {'error': f"Référence '{reference_name}' non trouvée – appelez set_reference() d'abord"}

        ref = self.reference_distributions[reference_name]
        drift_report = {
            'detected_at': datetime.now().isoformat(),
            'reference': reference_name,
            'features': {},
            'overall_drift': False,
            'drift_severity': 'none',
            'affected_features': []
        }

        feature_drift_scores = []
        for col in X_current.columns:
            if col not in ref:
                continue

            current_vals = X_current[col].dropna().values
            ref_hist = ref[col]['histogram']
            expected_vals = np.repeat(
                ref_hist[1][:-1],
                np.round(ref_hist[0] / ref_hist[0].sum() * len(current_vals)).astype(int)
            )

            if len(expected_vals) == 0 or len(current_vals) == 0:
                continue

            # PSI
            psi = self.compute_psi(expected_vals, current_vals)

            # Stats de base
            current_mean = float(X_current[col].mean())
            ref_mean = ref[col]['mean']
            ref_std = ref[col]['std']
            mean_shift = abs(current_mean - ref_mean) / ref_std if ref_std > 0 else 0

            has_drift = psi > self.PSI_THRESHOLD or mean_shift > 2.5
            drift_level = (
                'critical' if psi > 0.5 else
                'high' if psi > 0.3 else
                'medium' if psi > self.PSI_THRESHOLD else
                'low' if psi > 0.1 else 'none'
            )

            feature_report = {
                'psi': round(psi, 4),
                'mean_shift_sigma': round(mean_shift, 3),
                'current_mean': round(current_mean, 4),
                'reference_mean': round(ref_mean, 4),
                'drift_detected': has_drift,
                'drift_level': drift_level
            }

            drift_report['features'][col] = feature_report
            feature_drift_scores.append(psi)

            if has_drift:
                drift_report['affected_features'].append({
                    'feature': col,
                    'psi': round(psi, 4),
                    'severity': drift_level
                })

        # Dérive globale
        avg_psi = float(np.mean(feature_drift_scores)) if feature_drift_scores else 0
        drift_report['avg_psi'] = round(avg_psi, 4)
        drift_report['overall_drift'] = avg_psi > self.PSI_THRESHOLD or len(drift_report['affected_features']) >= 3
        drift_report['drift_severity'] = (
            'critical' if avg_psi > 0.5 else
            'high' if avg_psi > 0.3 else
            'medium' if avg_psi > 0.2 else
            'low' if avg_psi > 0.1 else 'none'
        )
        drift_report['retrain_recommended'] = drift_report['overall_drift']

        # Historique
        self.drift_history.append({
            'timestamp': datetime.now().isoformat(),
            'avg_psi': avg_psi,
            'n_drifted_features': len(drift_report['affected_features'])
        })

        if drift_report['retrain_recommended']:
            logger.warning(
                f"Dérive détectée! PSI moyen={avg_psi:.3f}, "
                f"{len(drift_report['affected_features'])} features affectées"
            )

        return drift_report

    def detect_prediction_drift(
        self, predictions_old: list, predictions_new: list
    ) -> dict:
        """
        Détecte si la distribution des prédictions a changé (concept drift).
        Utile pour détecter quand le modèle commence à se dégrader.
        """
        old = np.array(predictions_old)
        new = np.array(predictions_new)

        if len(old) == 0 or len(new) == 0:
            return {'error': 'Données insuffisantes'}

        psi = self.compute_psi(old, new)
        mean_shift = abs(old.mean() - new.mean())
        std_ratio = new.std() / (old.std() or 1e-10)

        return {
            'prediction_psi': round(psi, 4),
            'mean_shift': round(float(mean_shift), 4),
            'std_ratio': round(float(std_ratio), 4),
            'concept_drift_detected': psi > self.PSI_THRESHOLD or mean_shift > 0.1,
            'severity': 'high' if psi > 0.3 else 'medium' if psi > 0.2 else 'low',
            'recommendation': 'Ré-entraînement urgent' if psi > 0.3 else
                             'Surveillance accrue' if psi > 0.2 else 'Modèle stable'
        }

    def get_drift_trend(self, last_n: int = 10) -> dict:
        """Tendance de dérive sur les dernières analyses."""
        recent = self.drift_history[-last_n:]
        if not recent:
            return {'trend': 'no_data'}

        psii = [h['avg_psi'] for h in recent]
        trend = (
            'increasing' if len(psii) > 1 and psii[-1] > psii[0] * 1.2 else
            'decreasing' if len(psii) > 1 and psii[-1] < psii[0] * 0.8 else
            'stable'
        )

        return {
            'trend': trend,
            'current_avg_psi': round(psii[-1], 4) if psii else 0,
            'max_psi': round(max(psii), 4) if psii else 0,
            'history': recent
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. AUTO-RETRAIN SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════

class AutoRetrainScheduler:
    """
    Planification automatique du ré-entraînement des modèles.
    Déclencheurs : temporal (hebdomadaire), drift-based, performance-based.
    """

    def __init__(self, drift_detector: DriftDetector, registry: ModelRegistry):
        self.drift_detector = drift_detector
        self.registry = registry
        self.retrain_log: list = []
        self.next_scheduled_retrain: Optional[datetime] = None
        self._schedule_next()

    def _schedule_next(self):
        """Planifie le prochain ré-entraînement (tous les 7 jours)."""
        self.next_scheduled_retrain = datetime.now() + timedelta(days=7)

    def should_retrain(self, X_current: Optional[pd.DataFrame] = None,
                       current_auc: Optional[float] = None) -> dict:
        """
        Évalue si un ré-entraînement est nécessaire.
        Retourne la décision et les raisons.
        """
        reasons = []
        should = False

        # Déclencheur temporel
        if self.next_scheduled_retrain and datetime.now() >= self.next_scheduled_retrain:
            should = True
            reasons.append({
                'trigger': 'scheduled',
                'message': 'Ré-entraînement hebdomadaire planifié',
                'priority': 'medium'
            })

        # Déclencheur drift
        if X_current is not None and self.drift_detector.reference_distributions:
            drift = self.drift_detector.detect_drift(X_current)
            if drift.get('retrain_recommended'):
                should = True
                reasons.append({
                    'trigger': 'data_drift',
                    'message': f"Dérive des données détectée (PSI={drift.get('avg_psi', 0):.3f})",
                    'priority': 'high',
                    'details': drift
                })

        # Déclencheur performance
        if current_auc is not None and current_auc < 0.65:
            should = True
            reasons.append({
                'trigger': 'performance_degradation',
                'message': f"AUC en dessous du seuil: {current_auc:.3f} < 0.65",
                'priority': 'critical'
            })

        return {
            'should_retrain': should,
            'reasons': sorted(reasons, key=lambda x: {'critical': 0, 'high': 1, 'medium': 2}[x['priority']]),
            'next_scheduled': self.next_scheduled_retrain.isoformat() if self.next_scheduled_retrain else None,
            'evaluated_at': datetime.now().isoformat()
        }

    def execute_retrain(self, orchestrator_instance, users: list, tasks: list,
                        messages: list, leaves: list, activities: list,
                        feedbacks: list, conversations: list) -> dict:
        """
        Exécute le ré-entraînement complet et met à jour le registry.
        """
        start_time = datetime.now()
        version = start_time.strftime('v%Y%m%d_%H%M')

        logger.info(f"Démarrage ré-entraînement automatique {version}...")

        try:
            # Re-initialiser tous les modèles
            metrics = orchestrator_instance.initialize(
                users, tasks, messages, leaves,
                activities, feedbacks, conversations
            )

            duration = (datetime.now() - start_time).seconds

            # Enregistrer dans le registry
            self.registry.register_model(
                model_name='hr_ml_suite',
                version=version,
                metrics=metrics.get('turnover_model', {}),
                params={
                    'n_employees': metrics.get('n_employees'),
                    'n_feedbacks': len(feedbacks),
                    'n_tasks': len(tasks)
                },
                model_path=str(MODELS_DIR / f'model_{version}.pkl'),
                status='staging'
            )

            # Promouvoir si métriques suffisantes
            auc = metrics.get('turnover_model', {}).get('auc_mean', 0)
            if auc > 0.65 or metrics.get('turnover_model', {}).get('status') == 'synthetic':
                self.registry.promote_to_production('hr_ml_suite', version)
                production_status = 'promoted'
            else:
                production_status = 'kept_in_staging'

            # Mettre à jour la référence du drift detector
            X, _ = orchestrator_instance.feature_engine.build_feature_matrix(
                users, tasks, messages, leaves, activities, feedbacks
            )
            if not X.empty:
                self.drift_detector.set_reference(X, 'training')

            self._schedule_next()

            result = {
                'status': 'success',
                'version': version,
                'duration_seconds': duration,
                'metrics': metrics,
                'production_status': production_status,
                'next_scheduled': self.next_scheduled_retrain.isoformat()
            }

        except Exception as e:
            result = {
                'status': 'failed',
                'version': version,
                'error': str(e),
                'duration_seconds': (datetime.now() - start_time).seconds
            }
            logger.error(f"Ré-entraînement échoué: {e}", exc_info=True)

        self.retrain_log.append(result)
        return result

    def get_retrain_history(self, last_n: int = 10) -> list:
        return self.retrain_log[-last_n:]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. A/B TESTING — Comparaison de modèles en production
# ═══════════════════════════════════════════════════════════════════════════════

class ABTestingEngine:
    """
    Permet de tester deux versions de modèle en production simultanément.
    Split traffic: 80% modèle A (champion), 20% modèle B (challenger).
    Évalue lequel performe mieux et décide automatiquement.
    """

    def __init__(self):
        self.active_tests: dict = {}
        self.results: dict = defaultdict(list)

    def start_test(
        self, test_name: str,
        model_a_name: str, model_b_name: str,
        split_ratio: float = 0.2,
        success_metric: str = 'accuracy'
    ) -> dict:
        """Lance un test A/B entre deux modèles."""
        self.active_tests[test_name] = {
            'model_a': model_a_name,
            'model_b': model_b_name,
            'split_ratio': split_ratio,
            'success_metric': success_metric,
            'started_at': datetime.now().isoformat(),
            'status': 'active'
        }
        return self.active_tests[test_name]

    def assign_model(self, test_name: str, request_id: str) -> str:
        """
        Assigne un modèle à une requête (déterministe basé sur le hash).
        Retourne 'model_a' ou 'model_b'.
        """
        if test_name not in self.active_tests:
            return 'model_a'

        test = self.active_tests[test_name]
        # Déterministe : même request_id → même modèle
        hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        threshold = int(test['split_ratio'] * 100)

        return 'model_b' if (hash_val % 100) < threshold else 'model_a'

    def record_outcome(self, test_name: str, model: str, prediction: float,
                       outcome: Optional[float] = None, latency_ms: float = 0):
        """Enregistre le résultat d'une prédiction pour analyse."""
        self.results[test_name].append({
            'model': model,
            'prediction': prediction,
            'outcome': outcome,
            'latency_ms': latency_ms,
            'recorded_at': datetime.now().isoformat()
        })

    def analyze_test(self, test_name: str) -> dict:
        """Analyse statistique des résultats du test A/B."""
        if test_name not in self.active_tests:
            return {'error': 'Test introuvable'}

        results = self.results.get(test_name, [])
        if len(results) < 50:
            return {
                'test_name': test_name,
                'status': 'insufficient_data',
                'n_samples': len(results),
                'min_required': 50
            }

        a_results = [r for r in results if r['model'] == 'model_a']
        b_results = [r for r in results if r['model'] == 'model_b']

        # Métriques de base
        a_preds = [r['prediction'] for r in a_results]
        b_preds = [r['prediction'] for r in b_results]

        a_latency = np.mean([r['latency_ms'] for r in a_results])
        b_latency = np.mean([r['latency_ms'] for r in b_results])

        # Test de significativité (Welch's t-test simplifié)
        if len(a_preds) >= 5 and len(b_preds) >= 5:
            a_arr, b_arr = np.array(a_preds), np.array(b_preds)
            t_stat = abs(a_arr.mean() - b_arr.mean()) / np.sqrt(
                a_arr.var() / len(a_arr) + b_arr.var() / len(b_arr) + 1e-10
            )
            significant = t_stat > 1.96  # p < 0.05
        else:
            t_stat, significant = 0, False

        test = self.active_tests[test_name]
        winner = None
        if significant:
            # Dans le contexte RH, on veut la meilleure précision ET la plus faible latence
            winner = 'model_b' if (
                abs(np.mean(b_preds) - 0.5) < abs(np.mean(a_preds) - 0.5)
                and b_latency <= a_latency * 1.2
            ) else 'model_a'

        return {
            'test_name': test_name,
            'status': 'concluded' if significant and len(results) >= 200 else 'running',
            'n_total': len(results),
            'model_a': {
                'name': test['model_a'],
                'n_samples': len(a_results),
                'avg_prediction': round(float(np.mean(a_preds)), 4) if a_preds else 0,
                'avg_latency_ms': round(float(a_latency), 1)
            },
            'model_b': {
                'name': test['model_b'],
                'n_samples': len(b_results),
                'avg_prediction': round(float(np.mean(b_preds)), 4) if b_preds else 0,
                'avg_latency_ms': round(float(b_latency), 1)
            },
            'statistical_significance': significant,
            't_statistic': round(float(t_stat), 4),
            'winner': winner,
            'recommendation': (
                f"Promouvoir {test[winner]} en production" if winner else
                'Continuer le test – données insuffisantes pour conclure'
            )
        }

    def stop_test(self, test_name: str) -> dict:
        if test_name in self.active_tests:
            self.active_tests[test_name]['status'] = 'stopped'
            self.active_tests[test_name]['stopped_at'] = datetime.now().isoformat()
            return self.active_tests[test_name]
        return {'error': 'Test introuvable'}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MONITORING EN TEMPS RÉEL — Métriques de production
# ═══════════════════════════════════════════════════════════════════════════════

class MLMonitor:
    """
    Monitoring des modèles en production.
    Suit : latence, débit, erreurs, distribution des prédictions.
    Génère des alertes en cas d'anomalie.
    """

    def __init__(self, window_minutes: int = 60):
        self.window = timedelta(minutes=window_minutes)
        self.metrics_log: list = []
        self.alert_thresholds = {
            'max_latency_ms': 500,
            'max_error_rate': 0.05,
            'min_requests_per_hour': 1
        }

    def record_prediction(
        self, model_name: str, latency_ms: float,
        prediction: float, success: bool = True
    ):
        self.metrics_log.append({
            'model': model_name,
            'latency_ms': latency_ms,
            'prediction': prediction,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })
        # Garder seulement les 1000 dernières entrées
        if len(self.metrics_log) > 1000:
            self.metrics_log = self.metrics_log[-1000:]

    def get_current_metrics(self) -> dict:
        """Métriques de la dernière heure."""
        cutoff = datetime.now() - self.window
        recent = [
            m for m in self.metrics_log
            if datetime.fromisoformat(m['timestamp']) > cutoff
        ]

        if not recent:
            return {'status': 'no_data', 'window_minutes': self.window.seconds // 60}

        by_model: dict = defaultdict(list)
        for m in recent:
            by_model[m['model']].append(m)

        metrics = {}
        for model, logs in by_model.items():
            latencies = [l['latency_ms'] for l in logs]
            predictions = [l['prediction'] for l in logs]
            error_rate = sum(1 for l in logs if not l['success']) / len(logs)

            metrics[model] = {
                'requests': len(logs),
                'avg_latency_ms': round(float(np.mean(latencies)), 1),
                'p95_latency_ms': round(float(np.percentile(latencies, 95)), 1),
                'p99_latency_ms': round(float(np.percentile(latencies, 99)), 1),
                'error_rate': round(error_rate, 4),
                'avg_prediction': round(float(np.mean(predictions)), 4),
                'prediction_std': round(float(np.std(predictions)), 4)
            }

        return {
            'window_minutes': self.window.seconds // 60,
            'total_requests': len(recent),
            'by_model': metrics,
            'alerts': self._check_alerts(metrics)
        }

    def _check_alerts(self, metrics: dict) -> list:
        alerts = []
        for model, m in metrics.items():
            if m['p95_latency_ms'] > self.alert_thresholds['max_latency_ms']:
                alerts.append({
                    'model': model,
                    'type': 'high_latency',
                    'severity': 'high',
                    'message': f"Latence P95 élevée: {m['p95_latency_ms']}ms",
                    'threshold': self.alert_thresholds['max_latency_ms']
                })
            if m['error_rate'] > self.alert_thresholds['max_error_rate']:
                alerts.append({
                    'model': model,
                    'type': 'high_error_rate',
                    'severity': 'critical',
                    'message': f"Taux d'erreur: {m['error_rate']*100:.1f}%",
                    'threshold': self.alert_thresholds['max_error_rate']
                })
        return alerts


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES API MLOps
# ═══════════════════════════════════════════════════════════════════════════════

from flask import Blueprint, request, jsonify

mlops_bp = Blueprint('mlops', __name__, url_prefix='/api/mlops')

# Singletons
registry = ModelRegistry()
drift_detector = DriftDetector()
ab_engine = ABTestingEngine()
monitor = MLMonitor()
scheduler = AutoRetrainScheduler(drift_detector, registry)


@mlops_bp.route('/registry', methods=['GET'])
def get_registry():
    """GET /api/mlops/registry — État du registre des modèles."""
    return jsonify(registry.get_summary())


@mlops_bp.route('/registry/<model_name>/history', methods=['GET'])
def get_model_history(model_name: str):
    """GET /api/mlops/registry/{model}/history — Historique des versions."""
    return jsonify({'model': model_name, 'history': registry.get_model_history(model_name)})


@mlops_bp.route('/registry/<model_name>/rollback', methods=['POST'])
def rollback_model(model_name: str):
    """POST /api/mlops/registry/{model}/rollback — Rollback version précédente."""
    result = registry.rollback(model_name)
    if result:
        return jsonify({'status': 'rolled_back', 'version': result})
    return jsonify({'error': 'Aucune version archivée disponible'}), 404


@mlops_bp.route('/drift/detect', methods=['POST'])
def detect_drift():
    """
    POST /api/mlops/drift/detect
    Analyse la dérive des données actuelles.
    """
    try:
        from ml_engine import orchestrator
        from flask import current_app

        users = current_app.config.get('users', [])
        tasks = current_app.config.get('tasks', [])
        messages = current_app.config.get('messages', [])
        leaves = current_app.config.get('leaves', [])
        activities = current_app.config.get('activities', [])
        feedbacks = current_app.config.get('feedbacks', [])

        X, _ = orchestrator.feature_engine.build_feature_matrix(
            users, tasks, messages, leaves, activities, feedbacks
        )

        if X.empty:
            return jsonify({'error': 'Données insuffisantes'}), 400

        if not drift_detector.reference_distributions:
            drift_detector.set_reference(X, 'training')
            return jsonify({'message': 'Référence initialisée', 'n_features': len(X.columns)})

        report = drift_detector.detect_drift(X)
        trend = drift_detector.get_drift_trend()

        return jsonify({**report, 'drift_trend': trend})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@mlops_bp.route('/retrain/check', methods=['GET'])
def check_retrain():
    """GET /api/mlops/retrain/check — Vérifie si un ré-entraînement est nécessaire."""
    decision = scheduler.should_retrain()
    return jsonify(decision)


@mlops_bp.route('/retrain/execute', methods=['POST'])
def execute_retrain():
    """POST /api/mlops/retrain/execute — Lance un ré-entraînement."""
    try:
        from ml_engine import orchestrator
        from flask import current_app

        users = current_app.config.get('users', [])
        tasks = current_app.config.get('tasks', [])
        messages = current_app.config.get('messages', [])
        leaves = current_app.config.get('leaves', [])
        activities = current_app.config.get('activities', [])
        feedbacks = current_app.config.get('feedbacks', [])
        conversations = current_app.config.get('conversations', [])

        result = scheduler.execute_retrain(
            orchestrator, users, tasks, messages,
            leaves, activities, feedbacks, conversations
        )
        return jsonify(result)

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@mlops_bp.route('/retrain/history', methods=['GET'])
def get_retrain_history():
    """GET /api/mlops/retrain/history — Historique des ré-entraînements."""
    return jsonify(scheduler.get_retrain_history())


@mlops_bp.route('/ab-test', methods=['POST'])
def start_ab_test():
    """POST /api/mlops/ab-test — Lance un test A/B."""
    data = request.get_json()
    result = ab_engine.start_test(
        test_name=data.get('test_name', 'default_test'),
        model_a_name=data.get('model_a', 'production'),
        model_b_name=data.get('model_b', 'candidate'),
        split_ratio=float(data.get('split_ratio', 0.2))
    )
    return jsonify(result)


@mlops_bp.route('/ab-test/<test_name>/results', methods=['GET'])
def get_ab_results(test_name: str):
    """GET /api/mlops/ab-test/{name}/results — Résultats du test A/B."""
    return jsonify(ab_engine.analyze_test(test_name))


@mlops_bp.route('/monitoring', methods=['GET'])
def get_monitoring():
    """GET /api/mlops/monitoring — Métriques de monitoring en temps réel."""
    return jsonify(monitor.get_current_metrics())


@mlops_bp.route('/health', methods=['GET'])
def mlops_health():
    """GET /api/mlops/health — Vue synthétique de la santé MLOps."""
    return jsonify({
        'status': 'operational',
        'components': {
            'model_registry': 'ok',
            'drift_detector': 'ok' if drift_detector.reference_distributions else 'not_initialized',
            'ab_testing': f"{len(ab_engine.active_tests)} test(s) actif(s)",
            'monitor': 'ok',
            'scheduler': {
                'next_retrain': scheduler.next_scheduled_retrain.isoformat() if scheduler.next_scheduled_retrain else None
            }
        },
        'registry_summary': registry.get_summary(),
        'checked_at': datetime.now().isoformat()
    })