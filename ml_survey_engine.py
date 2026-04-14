"""
ml_survey_engine.py — Moteur IA d'analyse des enquêtes RH
══════════════════════════════════════════════════════════════════════════════
Stack : VADER+FR · TF-IDF · Statistiques avancées · NPS · Cronbach · Entropie

CORRECTIONS v2 :
  • safe_parse_json() : parse automatiquement les strings JSON (SQLite)
  • answers et questions normalisés avant tout traitement
  • max_scale accepte None, string, int, float
  • department None → "Non défini"
  • answers peut contenir des entiers directement (ex: scale=5)
  • Tous les accès dict sécurisés avec .get() + valeur par défaut
══════════════════════════════════════════════════════════════════════════════
pip install vaderSentiment scikit-learn numpy
"""

import re
import json
import math
import logging
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Imports optionnels ────────────────────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        HAS_VADER = True
    except ImportError:
        HAS_VADER = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES DE NORMALISATION (correction principale)
# ══════════════════════════════════════════════════════════════════════════════

def safe_parse_json(val: Any, default: Any) -> Any:
    """
    Parse une valeur JSON si c'est une string, sinon retourne tel quel.
    SQLite stocke les champs JSON comme strings — ce helper corrige ça.
    """
    if val is None:
        return default
    if isinstance(val, str):
        stripped = val.strip()
        if not stripped:
            return default
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return default
    return val


def safe_float(val: Any, default: float = 0.0) -> Optional[float]:
    """Convertit n'importe quel type en float, None si impossible."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def safe_max_scale(val: Any) -> float:
    """
    Extrait max_scale depuis un champ question.
    Accepte None, string "5", int 5, float 5.0 → retourne toujours un float >= 2.
    """
    f = safe_float(val)
    if f is None or f < 2:
        return 5.0
    return f


def normalize_survey(s: dict) -> dict:
    """Normalise un dict enquête retourné par SQLAlchemy to_dict()."""
    s = dict(s)
    s["questions"] = safe_parse_json(s.get("questions"), [])
    if not isinstance(s["questions"], list):
        s["questions"] = []
    return s


def normalize_response(r: dict) -> dict:
    """Normalise un dict SurveyResponse retourné par SQLAlchemy to_dict()."""
    r = dict(r)
    r["answers"]    = safe_parse_json(r.get("answers"), [])
    if not isinstance(r["answers"], list):
        r["answers"] = []
    r["department"] = r.get("department") or "Non défini"
    return r


# ══════════════════════════════════════════════════════════════════════════════
# LEXIQUES FRANÇAIS
# ══════════════════════════════════════════════════════════════════════════════

FR_POSITIVE_LEXICON: Dict[str, float] = {
    "excellent": 3.2, "parfait": 3.0, "super": 2.8, "génial": 2.8,
    "fantastique": 3.2, "formidable": 3.0, "remarquable": 2.8,
    "exceptionnel": 3.4, "extraordinaire": 3.2, "admirable": 2.6,
    "bien": 1.8, "bon": 2.0, "bonne": 2.0, "bonnes": 2.0, "bons": 2.0,
    "satisfait": 2.2, "content": 2.0, "heureux": 2.5, "ravi": 2.8,
    "enchanté": 2.8, "satisfaisant": 2.0, "motivé": 2.5, "engagé": 2.3,
    "dynamique": 2.3, "enthousiaste": 2.8, "passionné": 2.8, "investi": 2.3,
    "positif": 2.2, "optimiste": 2.3, "confiant": 2.0, "efficace": 2.3,
    "professionnel": 2.2, "compétent": 2.3, "qualité": 2.0, "performant": 2.3,
    "productif": 2.0, "rigoureux": 2.0, "fiable": 2.0, "organisé": 2.0,
    "sympa": 2.2, "convivial": 2.5, "chaleureux": 2.5, "agréable": 2.3,
    "solidaire": 2.5, "bienveillant": 2.5, "respectueux": 2.3, "équitable": 2.5,
    "transparent": 2.3, "ouvert": 1.8, "accessible": 2.0, "coopératif": 2.2,
    "amélioration": 1.8, "progrès": 2.0, "évolution": 1.8, "développement": 1.8,
    "apprentissage": 1.8, "enrichissant": 2.5, "stimulant": 2.5, "épanouissant": 2.8,
    "intéressant": 2.0, "valorisant": 2.5, "reconnu": 2.3, "valorisé": 2.5,
    "merci": 2.0, "bravo": 2.8, "félicitations": 3.0, "impeccable": 3.0,
    "irréprochable": 2.8, "idéal": 2.8, "parfaitement": 2.8, "adore": 2.8,
    "j'adore": 2.8, "aime": 2.0, "j'aime": 2.0, "love": 2.5,
}

FR_NEGATIVE_LEXICON: Dict[str, float] = {
    "mauvais": -2.5, "mal": -2.0, "médiocre": -2.5, "nul": -2.8,
    "décevant": -2.5, "insuffisant": -2.3, "inadmissible": -3.0,
    "inacceptable": -3.0, "catastrophique": -3.5, "terrible": -3.2,
    "horrible": -3.5, "désastreux": -3.5, "lamentable": -3.0,
    "problème": -1.8, "difficile": -1.5, "impossible": -2.5, "bloqué": -2.0,
    "retard": -1.8, "échec": -2.8, "raté": -2.5, "dysfonction": -2.8,
    "défaillant": -2.8, "désorganisé": -2.5, "chaotique": -2.8, "flou": -1.5,
    "surchargé": -2.5, "débordé": -2.5, "épuisé": -3.0, "burnout": -3.5,
    "pression": -2.0, "stress": -2.0, "stressé": -2.5, "surcharge": -2.5,
    "conflit": -2.5, "tension": -2.0, "injuste": -2.8, "inéquitable": -2.8,
    "favoritisme": -3.0, "discrimination": -3.5, "harcèlement": -3.8,
    "mépris": -3.0, "irrespect": -3.0, "manque": -2.0, "insuffisance": -2.3,
    "absence": -1.8, "jamais": -1.5, "rien": -2.0,
    "mécontent": -2.5, "insatisfait": -2.5, "déçu": -2.5, "frustré": -2.8,
    "démotivé": -2.8, "découragé": -2.8, "anxieux": -2.3, "inquiet": -2.0,
    "préoccupant": -2.0, "alarmant": -2.8, "grave": -2.3, "critique": -1.5,
    "incompétent": -3.0, "inefficace": -2.5, "opaque": -2.0, "fermé": -1.8,
}

FR_NEGATIONS = frozenset({
    "ne", "n'", "pas", "plus", "jamais", "rien", "aucun", "non", "sans", "ni",
})

FR_INTENSIFIERS = {
    "très": 1.4, "vraiment": 1.3, "tellement": 1.4, "extrêmement": 1.6,
    "absolument": 1.5, "totalement": 1.4, "complètement": 1.4,
    "particulièrement": 1.2, "surtout": 1.1, "peu": 0.5, "assez": 0.9, "plutôt": 0.85,
}

SARCASM_PATTERNS = [
    r"bien sûr[!.]{2,}", r"évidemment[!.]{2,}", r"comme d'habitude",
    r"comme toujours", r"c'est (tellement |vraiment )?super[.!]{2,}",
    r"oh super", r"quelle surprise", r"génial\.{2,}",
]

TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "charge_travail": ["charge", "travail", "heures", "surcharge", "débordé", "temps",
                       "délai", "urgence", "pression", "rythme", "volume", "tâches",
                       "surchargé", "épuisement", "deadline"],
    "management":     ["manager", "chef", "responsable", "direction", "supérieur",
                       "hiérarchie", "encadrement", "leadership", "autorité", "décision",
                       "patron", "directeur", "management", "coaching"],
    "ambiance":       ["ambiance", "équipe", "collègue", "atmosphère", "cohésion",
                       "solidarité", "entente", "relation", "esprit", "groupe",
                       "convivialité", "camaraderie", "collaboration", "entraide"],
    "rémunération":   ["salaire", "rémunération", "prime", "compensation", "augmentation",
                       "avantage", "paye", "revenu", "indemnité", "bonus", "argent"],
    "formation":      ["formation", "compétence", "apprentissage", "développement",
                       "évolution", "carrière", "montée en compétence", "certification"],
    "organisation":   ["organisation", "processus", "procédure", "outil", "ressource",
                       "planning", "workflow", "structure", "méthode", "règle"],
    "bien_etre":      ["bien-être", "santé", "équilibre", "stress", "repos", "confort",
                       "sécurité", "soutien", "bien être", "burnout", "fatigue"],
    "communication":  ["communication", "information", "transparence", "réunion",
                       "feedback", "retour", "échange", "partage", "dialogue", "clarté"],
    "reconnaissance": ["reconnaissance", "valorisation", "appréciation", "mérite",
                       "récompense", "considération", "respect", "gratitude", "valeur"],
}


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSEUR DE SENTIMENT FRANÇAIS
# ══════════════════════════════════════════════════════════════════════════════

class FrenchSentimentAnalyzer:
    def __init__(self):
        self.vader = None
        self._init_vader()

    def _init_vader(self):
        if not HAS_VADER:
            return
        try:
            self.vader = SentimentIntensityAnalyzer()
            for word, score in FR_POSITIVE_LEXICON.items():
                self.vader.lexicon[word] = score
                self.vader.lexicon[word.capitalize()] = score
            for word, score in FR_NEGATIVE_LEXICON.items():
                self.vader.lexicon[word] = score
                self.vader.lexicon[word.capitalize()] = score
        except Exception as e:
            logger.warning("Erreur init VADER : %s", e)
            self.vader = None

    def _fr_lexicon_score(self, text: str) -> Tuple[float, int]:
        text_lower = text.lower()
        tokens = re.findall(r"\b\w[\w']*\b", text_lower)
        if not tokens:
            return 0.0, 0
        scores = []
        fr_word_count = 0
        prev_negation = False
        prev_intensifier = 1.0
        for token in tokens:
            if token in FR_NEGATIONS:
                prev_negation = True
                continue
            if token in FR_INTENSIFIERS:
                prev_intensifier *= FR_INTENSIFIERS[token]
                continue
            score = None
            if token in FR_POSITIVE_LEXICON:
                score = FR_POSITIVE_LEXICON[token]
                fr_word_count += 1
            elif token in FR_NEGATIVE_LEXICON:
                score = FR_NEGATIVE_LEXICON[token]
                fr_word_count += 1
            if score is not None:
                if prev_negation:
                    score = -score * 0.6
                score *= prev_intensifier
                scores.append(max(-4, min(4, score)))
            prev_negation = False
            prev_intensifier = 1.0
        if not scores:
            return 0.0, fr_word_count
        raw = sum(scores) / (len(scores) + 1)
        return max(-1.0, min(1.0, raw)), fr_word_count

    def _detect_sarcasm(self, text: str, compound: float) -> bool:
        text_lower = text.lower()
        for pattern in SARCASM_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        excessive_punct = len(re.findall(r"[!?]{2,}", text)) >= 2
        overly_positive = compound > 0.4
        contains_doubt  = any(w in text_lower for w in ["mais", "cependant", "pourtant", "quand même"])
        if excessive_punct and overly_positive and contains_doubt:
            return True
        upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if upper_ratio > 0.4 and compound > 0.3:
            return True
        return False

    def _detect_topics(self, text: str) -> List[str]:
        text_lower = text.lower()
        return [t for t, kws in TOPIC_KEYWORDS.items() if any(kw in text_lower for kw in kws)]

    def _fr_ratio(self, text: str) -> float:
        tokens = re.findall(r"\b\w{3,}\b", text.lower())
        if not tokens:
            return 0.0
        fr = sum(1 for t in tokens if t in FR_POSITIVE_LEXICON or t in FR_NEGATIVE_LEXICON or t in FR_NEGATIONS)
        return min(1.0, fr / len(tokens) * 4)

    def analyze(self, text: str) -> Dict[str, Any]:
        if not text or not str(text).strip():
            return {
                "compound_score": 0.0, "label": "neutral", "confidence": 0.5,
                "intensity": 0.0, "topics": [], "sarcasm_detected": False,
            }
        text_clean = str(text).strip()
        fr_ratio = self._fr_ratio(text_clean)

        vader_compound = 0.0
        if self.vader:
            try:
                vs = self.vader.polarity_scores(text_clean)
                vader_compound = vs.get("compound", 0.0)
            except Exception:
                pass

        fr_compound, _ = self._fr_lexicon_score(text_clean)

        w_vader = 1.0 - fr_ratio * 0.6
        w_fr    = fr_ratio * 0.6
        compound = (vader_compound * w_vader + fr_compound * w_fr) if self.vader else fr_compound
        compound = max(-1.0, min(1.0, compound))

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        confidence = max(0.30, min(0.99, 0.5 + abs(compound) * 0.45))
        sarcasm    = self._detect_sarcasm(text_clean, compound)
        topics     = self._detect_topics(text_clean)

        return {
            "compound_score":   round(compound, 4),
            "label":            label,
            "confidence":       round(confidence, 4),
            "intensity":        round(abs(compound), 4),
            "topics":           topics,
            "sarcasm_detected": sarcasm,
        }


# ══════════════════════════════════════════════════════════════════════════════
# STATISTIQUES AVANCÉES
# ══════════════════════════════════════════════════════════════════════════════

def compute_skewness(nums: List[float]) -> float:
    n = len(nums)
    if n < 3:
        return 0.0
    mean = statistics.mean(nums)
    std  = statistics.stdev(nums)
    if std == 0:
        return 0.0
    return sum(((x - mean) / std) ** 3 for x in nums) * n / ((n-1) * (n-2))


def compute_kurtosis(nums: List[float]) -> float:
    n = len(nums)
    if n < 4:
        return 0.0
    mean = statistics.mean(nums)
    std  = statistics.stdev(nums)
    if std == 0:
        return 0.0
    return (sum(((x - mean) / std) ** 4 for x in nums) * n * (n+1)
            / ((n-1) * (n-2) * (n-3))
            - 3 * (n-1)**2 / ((n-2) * (n-3)))


def compute_entropy(counts: List[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c/total) * math.log2(c/total) for c in counts if c > 0)


def compute_percentiles(nums: List[float]) -> Dict[str, float]:
    if not nums:
        return {}
    s = sorted(nums)
    n = len(s)
    def pct(p):
        idx = (n - 1) * p / 100
        lo  = int(idx)
        hi  = min(lo + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)
    return {k: round(pct(v), 2) for k, v in [("p10",10),("p25",25),("p50",50),("p75",75),("p90",90)]}


def compute_cronbach_alpha(items: List[List[float]]) -> Optional[float]:
    if not HAS_NUMPY or len(items) < 2:
        return None
    try:
        import numpy as np
        matrix = np.array(items, dtype=float).T
        if matrix.shape[0] < 2:
            return None
        k = matrix.shape[1]
        item_variances = matrix.var(axis=0, ddof=1)
        total_variance = matrix.sum(axis=1).var(ddof=1)
        if total_variance == 0:
            return None
        alpha = k / (k - 1) * (1 - item_variances.sum() / total_variance)
        return round(float(np.clip(alpha, -1, 1)), 3)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MOTEUR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class SurveyMLEngine:
    def __init__(self):
        self.sentiment_analyzer = FrenchSentimentAnalyzer()
        self.tfidf = None
        if HAS_SKLEARN:
            try:
                self.tfidf = TfidfVectorizer(
                    max_features=150, ngram_range=(1, 2),
                    min_df=1, sublinear_tf=True,
                )
            except Exception:
                pass

    # ── API publique ──────────────────────────────────────────────────────

    def analyze_survey(self, survey_dict: dict, responses_list: list) -> dict:
        """Analyse NLP + stats complète d'une enquête."""
        # ▶ CORRECTION : normaliser avant tout traitement
        survey    = normalize_survey(survey_dict)
        responses = [normalize_response(r) for r in responses_list]

        questions  = survey.get("questions", [])
        total_resp = len(responses)
        analyzed_qs = []

        for i, question in enumerate(questions):
            answers = self._extract_answers(responses, i)
            qa = self._analyze_question(question, answers, total_resp, i)
            analyzed_qs.append(qa)

        # Score de santé global
        health_scores = []
        for qa in analyzed_qs:
            if "ml_analysis" in qa:
                h = qa["ml_analysis"].get("health_score")
                if h is not None:
                    health_scores.append(h)
            elif "stats" in qa:
                h = qa["stats"].get("normalized_score")
                if h is not None:
                    health_scores.append(h)

        overall_health = round(statistics.mean(health_scores), 1) if health_scores else None

        # Alpha de Cronbach
        scale_answers = []
        for i, qa in enumerate(analyzed_qs):
            if "stats" in qa and qa["stats"].get("mean") is not None:
                nums = []
                for r in responses:
                    ans = r.get("answers", [])
                    if i < len(ans):
                        v = safe_float(ans[i])
                        if v is not None:
                            nums.append(v)
                if nums:
                    scale_answers.append(nums)

        cronbach = compute_cronbach_alpha(scale_answers) if len(scale_answers) >= 2 else None
        dept_breakdown = self._compute_dept_breakdown(responses, analyzed_qs, questions)
        recommendations = self._generate_recommendations(analyzed_qs, overall_health)

        return {
            "survey":               survey_dict,   # retourner l'original (pas modifié)
            "total_responses":      total_resp,
            "overall_health_score": overall_health,
            "cronbach_alpha":       cronbach,
            "questions_analysis":   analyzed_qs,
            "department_breakdown": dept_breakdown,
            "recommendations":      recommendations,
            "analyzed_at":          datetime.now().isoformat(),
        }

    def get_overview(self, surveys_data: list) -> list:
        """Vue d'ensemble ML rapide."""
        overview = []
        for item in surveys_data:
            survey    = normalize_survey(item["survey"])
            responses = [normalize_response(r) for r in item.get("responses", [])]
            questions = survey.get("questions", [])
            total     = len(responses)

            text_scores  = []
            scale_scores = []

            for r in responses:
                for i, q in enumerate(questions):
                    if i >= len(r.get("answers", [])):
                        continue
                    ans = r["answers"][i]
                    qtype = q.get("type", "")

                    if qtype == "text" and ans and str(ans).strip():
                        res = self.sentiment_analyzer.analyze(str(ans))
                        text_scores.append(res["compound_score"])

                    elif qtype == "scale" and ans is not None:
                        v       = safe_float(ans)
                        max_val = safe_max_scale(q.get("max"))
                        if v is not None and max_val > 1:
                            scale_scores.append(v / (max_val - 1) - 1 / (max_val - 1))

            components = []
            if text_scores:
                components.append((statistics.mean(text_scores) + 1) / 2 * 100)
            if scale_scores:
                components.append((statistics.mean(scale_scores) + 1) / 2 * 100)

            health = round(statistics.mean(components), 1) if components else None
            nps    = round((statistics.mean(text_scores) if text_scores else 0) * 100, 1)

            overview.append({
                "id":                  item["survey"]["id"],
                "title":               survey.get("title", ""),
                "description":         survey.get("description", ""),
                "status":              survey.get("status", "active"),
                "target_department":   survey.get("target_department", "all"),
                "anonymous":           survey.get("anonymous", False),
                "created_at":          survey.get("created_at"),
                "deadline":            survey.get("deadline"),
                "total_responses":     total,
                "question_count":      len(questions),
                "ml_health_score":     health,
                "ml_nps":              nps if text_scores else None,
                "has_text_questions":  any(q.get("type") == "text"  for q in questions),
                "has_scale_questions": any(q.get("type") == "scale" for q in questions),
            })

        overview.sort(key=lambda x: x["created_at"] or "", reverse=True)
        return overview

    def get_global_insights(self, surveys_data: list) -> dict:
        """Insights ML agrégés sur toutes les enquêtes."""
        all_text_scores: List[float]              = []
        all_topics:      Counter                   = Counter()
        dept_scores:     Dict[str, List[float]]    = defaultdict(list)
        timeline:        List[dict]                = []

        for item in surveys_data:
            survey    = normalize_survey(item["survey"])
            responses = [normalize_response(r) for r in item.get("responses", [])]
            questions = survey.get("questions", [])
            created   = (survey.get("created_at") or "")[:10]
            survey_scores: List[float] = []

            for r in responses:
                dept = r.get("department", "N/A")
                for i, q in enumerate(questions):
                    if q.get("type") == "text" and i < len(r.get("answers", [])):
                        ans = r["answers"][i]
                        if ans and str(ans).strip():
                            res = self.sentiment_analyzer.analyze(str(ans))
                            sc  = res["compound_score"]
                            all_text_scores.append(sc)
                            survey_scores.append(sc)
                            dept_scores[dept].append(sc)
                            for t in res["topics"]:
                                all_topics[t] += 1

            if survey_scores and created:
                avg = statistics.mean(survey_scores)
                timeline.append({
                    "date":   created,
                    "title":  survey.get("title", ""),
                    "score":  round((avg + 1) / 2 * 100, 1),
                    "n_resp": len(responses),
                })

        dept_health = {
            dept: round((statistics.mean(scores) + 1) / 2 * 100, 1)
            for dept, scores in dept_scores.items() if scores
        }
        global_avg    = statistics.mean(all_text_scores) if all_text_scores else None
        global_health = round((global_avg + 1) / 2 * 100, 1) if global_avg is not None else None
        global_nps    = round((global_avg or 0) * 100, 1)

        timeline_sorted = sorted(timeline, key=lambda x: x["date"])
        if len(timeline_sorted) >= 2:
            trend = ("improving" if timeline_sorted[-1]["score"] > timeline_sorted[0]["score"]
                     else "declining" if timeline_sorted[-1]["score"] < timeline_sorted[0]["score"]
                     else "stable")
        else:
            trend = "stable"

        return {
            "global_health_score":  global_health,
            "global_nps":           global_nps,
            "total_text_responses": len(all_text_scores),
            "top_topics":           all_topics.most_common(8),
            "department_health":    dept_health,
            "timeline":             timeline_sorted,
            "trend":                trend,
            "analyzed_at":          datetime.now().isoformat(),
        }

    # ── Extraction robuste des réponses ───────────────────────────────────

    def _extract_answers(self, responses: list, question_idx: int) -> list:
        """Extrait les réponses à une question donnée, filtre None et vide."""
        answers = []
        for r in responses:
            ans_list = r.get("answers", [])
            if question_idx < len(ans_list):
                val = ans_list[question_idx]
                # Accepter entiers, floats, strings non-vides — rejeter None et ""
                if val is not None and str(val).strip() != "":
                    answers.append(val)
        return answers

    # ── Analyse par type de question ──────────────────────────────────────

    def _analyze_question(self, question: dict, answers: list,
                           total_resp: int, index: int) -> dict:
        q_type = question.get("type") or "text"
        qa = {
            "index":          index,
            "label":          question.get("label") or f"Question {index+1}",
            "type":           q_type,
            "question":       question,
            "response_count": len(answers),
            "response_rate":  round(len(answers) / max(total_resp, 1) * 100, 1),
        }
        if q_type == "text":
            qa.update(self._analyze_text(answers))
        elif q_type == "scale":
            qa.update(self._analyze_scale(answers, question))
        elif q_type in ("single_choice", "multiple_choice"):
            qa.update(self._analyze_choice(answers))
        return qa

    def _analyze_text(self, answers: list) -> dict:
        """NLP multicouche sur réponses textuelles."""
        analyses     = []
        sent_counts  = Counter()
        all_topics   = Counter()
        intensities  = []
        scores_raw   = []

        for ans in answers:
            text = str(ans).strip()
            if not text:
                continue
            res = self.sentiment_analyzer.analyze(text)
            analyses.append({
                "text":      text[:300],
                "sentiment": res["label"],
                "confidence":res["confidence"],
                "intensity": res["intensity"],
                "topics":    res["topics"],
                "sarcasm":   res["sarcasm_detected"],
                "score":     res["compound_score"],
            })
            sent_counts[res["label"]] += 1
            for t in res["topics"]:
                all_topics[t] += 1
            intensities.append(res["intensity"])
            scores_raw.append(res["compound_score"])

        n    = len(analyses)
        dist = {k: round(v / max(n, 1), 3) for k, v in sent_counts.items()}
        pos  = dist.get("positive", 0)
        neg  = dist.get("negative", 0)
        nps  = round((pos - neg) * 100, 1)

        avg_score = statistics.mean(scores_raw) if scores_raw else 0.0
        health    = round((avg_score + 1) / 2 * 100, 1)
        std_score = statistics.stdev(scores_raw) if len(scores_raw) > 1 else 0.0
        percentiles = compute_percentiles(scores_raw) if scores_raw else {}

        keywords = self._extract_keywords([a["text"] for a in analyses])

        return {
            "ml_analysis": {
                "sentiment_distribution": dist,
                "nps_score":      nps,
                "health_score":   health,
                "avg_intensity":  round(statistics.mean(intensities), 3) if intensities else 0,
                "top_topics":     all_topics.most_common(5),
                "top_keywords":   keywords,
                "sarcasm_count":  sum(1 for a in analyses if a["sarcasm"]),
                "responses_detail": sorted(analyses, key=lambda x: -abs(x["score"]))[:20],
                "interpretation": self._interpret_nps(nps, dist, n),
                "stats": {
                    "avg_score":       round(avg_score, 3),
                    "std_score":       round(std_score, 3),
                    "percentiles":     percentiles,
                    "positive_count":  sent_counts.get("positive", 0),
                    "neutral_count":   sent_counts.get("neutral",  0),
                    "negative_count":  sent_counts.get("negative", 0),
                    "total_analyzed":  n,
                },
            }
        }

    def _analyze_scale(self, answers: list, question: dict) -> dict:
        """
        Statistiques descriptives avancées.
        ▶ CORRECTION : safe_float() pour tous les types (int, float, string)
                       safe_max_scale() pour max_scale None ou string
        """
        nums: List[float] = []
        for a in answers:
            v = safe_float(a)
            if v is not None:
                nums.append(v)

        if not nums:
            return {
                "stats": {
                    "mean": None, "median": None, "mode": None,
                    "std_dev": None, "min": None, "max": None,
                    "max_scale": int(safe_max_scale(question.get("max"))),
                    "n": 0, "normalized_score": None,
                    "benchmark_label": "N/A", "benchmark_color": "gray",
                    "value_distribution": {}, "value_distribution_pct": {},
                    "segments": {"low":{"count":0,"pct":0},"mid":{"count":0,"pct":0},"high":{"count":0,"pct":0}},
                    "message": "Aucune réponse numérique exploitable",
                }
            }

        max_val = safe_max_scale(question.get("max"))
        n       = len(nums)
        mean    = statistics.mean(nums)
        median  = statistics.median(nums)
        std     = statistics.stdev(nums) if n > 1 else 0.0
        mode_v  = Counter(round(x) for x in nums).most_common(1)[0][0]

        skew = compute_skewness(nums)
        kurt = compute_kurtosis(nums)
        pcts = compute_percentiles(nums)
        vd   = Counter(round(x) for x in nums)

        normalized = round((mean - 1) / max(max_val - 1, 1) * 100, 1)
        normalized = max(0.0, min(100.0, normalized))

        if normalized >= 80: bmark_label, bmark_color = "Excellent",    "green"
        elif normalized >= 65: bmark_label, bmark_color = "Bon",         "blue"
        elif normalized >= 50: bmark_label, bmark_color = "Moyen",       "yellow"
        elif normalized >= 35: bmark_label, bmark_color = "Insuffisant", "orange"
        else:                  bmark_label, bmark_color = "Critique",    "red"

        low_thr  = max_val * 0.33
        high_thr = max_val * 0.67
        low_count  = sum(1 for x in nums if x <= low_thr)
        mid_count  = sum(1 for x in nums if low_thr < x <= high_thr)
        high_count = sum(1 for x in nums if x > high_thr)

        total_votes = max(n, 1)
        dist_enriched = {
            str(val): {"count": cnt, "pct": round(cnt / total_votes * 100, 1)}
            for val, cnt in sorted(vd.items())
        }

        return {
            "stats": {
                "mean":      round(mean, 2),
                "median":    round(median, 2),
                "mode":      mode_v,
                "std_dev":   round(std, 2),
                "min":       min(nums),
                "max":       max(nums),
                "max_scale": int(max_val),
                "n":         n,
                "normalized_score":  normalized,
                "benchmark_label":   bmark_label,
                "benchmark_color":   bmark_color,
                "skewness":          round(skew, 3),
                "kurtosis":          round(kurt, 3),
                "percentiles":       pcts,
                "value_distribution":     {k: v["count"] for k, v in dist_enriched.items()},
                "value_distribution_pct": dist_enriched,
                "segments": {
                    "low":  {"count": low_count,  "pct": round(low_count  / n * 100, 1)},
                    "mid":  {"count": mid_count,   "pct": round(mid_count  / n * 100, 1)},
                    "high": {"count": high_count,  "pct": round(high_count / n * 100, 1)},
                },
            }
        }

    def _analyze_choice(self, answers: list) -> dict:
        """Distribution + entropie pour questions à choix."""
        opt_counter: Counter = Counter()
        for ans in answers:
            # ▶ CORRECTION : gérer les listes JSON imbriquées et les strings JSON
            opts = safe_parse_json(ans, ans)
            if not isinstance(opts, list):
                opts = [opts]
            for opt in opts:
                if opt is not None and str(opt).strip():
                    opt_counter[str(opt)] += 1

        total_votes = sum(opt_counter.values()) or 1
        opt_dist = {
            opt: {"count": cnt, "pct": round(cnt / total_votes * 100, 1)}
            for opt, cnt in opt_counter.most_common()
        }
        top = opt_counter.most_common(1)
        dominant_opt = top[0][0] if top else "—"
        dominant_cnt = top[0][1] if top else 0
        dominance    = round(dominant_cnt / total_votes * 100, 1)
        consensus    = "Forte" if dominance > 60 else "Modérée" if dominance > 40 else "Faible"
        entropy      = round(compute_entropy(list(opt_counter.values())), 3)

        return {
            "distribution":    opt_dist,
            "dominant_option": dominant_opt,
            "dominance_pct":   dominance,
            "consensus":       consensus,
            "unique_options":  len(opt_counter),
            "entropy":         entropy,
        }

    # ── Helpers internes ─────────────────────────────────────────────────

    def _extract_keywords(self, texts: List[str]) -> list:
        FR_STOP = {
            "le","la","les","de","du","des","un","une","et","en","est","que","qui","se",
            "il","elle","ils","elles","on","au","aux","par","pour","avec","dans","sur",
            "ce","ma","mon","son","sa","nos","vous","nous","leur","leurs","ou","je","tu",
            "me","te","lui","plus","très","bien","avoir","être","faire","tout","mais",
            "aussi","comme","si","ne","pas","car","donc","or","ni","cela","ça","cette",
            "ces","cet","dont","où","quand","comment","même","encore","autre","puis",
        }
        if HAS_SKLEARN and self.tfidf and len(texts) >= 2:
            try:
                clean = [re.sub(r"[^\w\s]", " ", t.lower()) for t in texts]
                matrix = self.tfidf.fit_transform(clean)
                scores = matrix.sum(axis=0).A1
                features = self.tfidf.get_feature_names_out()
                pairs = [
                    (feat, round(float(sc), 3))
                    for feat, sc in zip(features, scores)
                    if feat not in FR_STOP and len(feat) > 2
                ]
                pairs.sort(key=lambda x: -x[1])
                return pairs[:10]
            except Exception:
                pass
        counter: Counter = Counter()
        for text in texts:
            for w in re.findall(r"\b\w{3,}\b", text.lower()):
                if w not in FR_STOP:
                    counter[w] += 1
        return [(w, cnt) for w, cnt in counter.most_common(10)]

    def _compute_dept_breakdown(self, responses: list, analyzed_qs: list,
                                 questions: list) -> dict:
        dept_data: Dict[str, dict] = defaultdict(lambda: {"count": 0, "scores": []})
        for r in responses:
            dept = r.get("department", "Non défini")
            dept_data[dept]["count"] += 1
            for i, qa in enumerate(analyzed_qs):
                if i >= len(r.get("answers", [])):
                    continue
                ans = r["answers"][i]
                if ans is None or str(ans).strip() == "":
                    continue
                if qa["type"] == "text":
                    res = self.sentiment_analyzer.analyze(str(ans))
                    dept_data[dept]["scores"].append((res["compound_score"] + 1) / 2 * 100)
                elif qa["type"] == "scale":
                    v = safe_float(ans)
                    if v is not None:
                        q = questions[i] if i < len(questions) else {}
                        max_s = safe_max_scale(q.get("max"))
                        dept_data[dept]["scores"].append(v / max_s * 100)

        return {
            dept: {
                "count":        d["count"],
                "health_score": round(statistics.mean(d["scores"]), 1) if d["scores"] else None,
            }
            for dept, d in dept_data.items()
        }

    def _generate_recommendations(self, analyzed_qs: list,
                                   overall_health) -> list:
        recs = []
        if overall_health is not None and overall_health < 50:
            recs.append({
                "priority": "critical",
                "question": "Enquête globale",
                "insight":  f"Score de santé global insuffisant : {overall_health:.0f}/100",
                "action":   "Réunion de direction urgente — plan d'action immédiat avec suivi",
            })

        for qa in analyzed_qs:
            label = qa.get("label", "cette question")
            if "ml_analysis" in qa:
                ml  = qa["ml_analysis"]
                nps = ml.get("nps_score", 0)
                neg = ml.get("stats", {}).get("negative_count", 0)
                topics = ml.get("top_topics", [])

                if nps < -30:
                    recs.append({"priority": "critical", "question": label,
                                 "insight": f"NPS très négatif ({nps:+.0f}) — {neg} réponses négatives",
                                 "action":  "Entretiens individuels et groupe de travail urgents"})
                elif nps < 0:
                    recs.append({"priority": "high", "question": label,
                                 "insight": f"NPS négatif ({nps:+.0f}) — majorité d'avis défavorables",
                                 "action":  "Identifier les causes et lancer un plan correctif sous 30 jours"})
                elif nps < 20:
                    recs.append({"priority": "medium", "question": label,
                                 "insight": f"NPS modéré ({nps:+.0f}) — marge d'amélioration",
                                 "action":  "Valoriser les points positifs et traiter les irritants"})
                if topics:
                    t = topics[0]
                    tname = (t[0] if isinstance(t, (list, tuple)) else t).replace("_", " ")
                    cnt   = (f" ({t[1]} mentions)" if isinstance(t, (list, tuple)) else "")
                    recs.append({"priority": "medium", "question": label,
                                 "insight": f"Thème dominant NLP : « {tname} »{cnt}",
                                 "action":  f"Prioriser « {tname} » dans le plan d'action RH"})

            elif "stats" in qa:
                sc = qa["stats"].get("normalized_score")
                if sc is not None:
                    if sc < 35:
                        recs.append({"priority": "critical", "question": label,
                                     "insight": f"Score critique {sc:.0f}/100 — intervention urgente",
                                     "action":  "Analyse et plan correctif avec suivi hebdomadaire"})
                    elif sc < 50:
                        recs.append({"priority": "high", "question": label,
                                     "insight": f"Score insuffisant {sc:.0f}/100",
                                     "action":  "Actions ciblées et mesure d'impact à 3 mois"})
                    elif sc < 65:
                        recs.append({"priority": "medium", "question": label,
                                     "insight": f"Score moyen {sc:.0f}/100 — amélioration possible",
                                     "action":  "Ateliers collaboratifs pour identifier les axes d'amélioration"})

        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recs.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 3))
        return recs[:8]

    @staticmethod
    def _interpret_nps(nps: float, dist: dict, total: int) -> str:
        neg_pct = round(dist.get("negative", 0) * 100)
        pos_pct = round(dist.get("positive", 0) * 100)
        if total == 0:
            return "Aucune réponse analysée"
        if nps > 50:
            return f"Satisfaction très élevée (NPS {nps:+.0f}) — {pos_pct}% positif sur {total} réponses"
        if nps > 20:
            return f"Bonne satisfaction (NPS {nps:+.0f}) — quelques points d'amélioration possibles"
        if nps > 0:
            return f"Satisfaction mitigée (NPS {nps:+.0f}) — équilibre fragile positif/négatif"
        if nps > -20:
            return f"Insatisfaction modérée (NPS {nps:+.0f}) — {neg_pct}% négatif · plan correctif recommandé"
        return f"Insatisfaction critique (NPS {nps:+.0f}) — {neg_pct}% de réponses négatives · intervention urgente"


# ── Instance globale ──────────────────────────────────────────────────────
survey_ml_engine = SurveyMLEngine()