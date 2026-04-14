"""
seed_plans.py
─────────────
Script à exécuter UNE SEULE FOIS pour insérer les 3 plans
d'abonnement CommSight dans la base de données.

Usage :
  python seed_plans.py

Les plans correspondent exactement à PricingPage.jsx.
"""

from app import app
from models import db
from models_tenant import SubscriptionPlan

PLANS = [
    {
        'code':            'monthly',
        'name':            'Mensuel',
        'price_eur':       29.0,
        'billing_period':  'monthly',
        'max_employees':   50,
        'max_departments': 5,
        'max_storage_gb':  5,
        'features': [
            'tasks',
            'posts',
            'messaging',
            'leaves',
            'surveys',
            'analytics_basic',
        ],
    },
    {
        'code':            'annual',
        'name':            'Annuel',
        'price_eur':       19.0,          # facturé 228€/an
        'billing_period':  'yearly',
        'max_employees':   200,
        'max_departments': -1,            # illimité
        'max_storage_gb':  25,
        'features': [
            'tasks',
            'posts',
            'messaging',
            'leaves',
            'surveys',
            'analytics_basic',
            'analytics_advanced',
            'ai_analytics',
            'ai_sentiment',
            'ai_health_score',
            'ai_coach',
            'support_priority',
        ],
    },
    {
        'code':            'lifetime',
        'name':            'À Vie',
        'price_eur':       599.0,         # paiement unique
        'billing_period':  'once',
        'max_employees':   -1,            # illimité
        'max_departments': -1,
        'max_storage_gb':  100,
        'features': [
            'tasks',
            'posts',
            'messaging',
            'leaves',
            'surveys',
            'analytics_basic',
            'analytics_advanced',
            'ai_analytics',
            'ai_sentiment',
            'ai_health_score',
            'ai_coach',
            'api_access',
            'support_priority_24_7',
            'future_features',
        ],
    },
]

def seed():
    with app.app_context():
        db.create_all()

        inserted = 0
        updated  = 0

        for p in PLANS:
            existing = SubscriptionPlan.query.filter_by(code=p['code']).first()
            if existing:
                for k, v in p.items():
                    setattr(existing, k, v)
                updated += 1
                print(f"  ↺  Plan mis à jour : {p['name']} ({p['code']})")
            else:
                plan = SubscriptionPlan(**p)
                db.session.add(plan)
                inserted += 1
                print(f"  ✅ Plan créé       : {p['name']} ({p['code']}) — {p['price_eur']}€/{p['billing_period']}")

        db.session.commit()
        print(f"\n✔  Terminé — {inserted} créé(s), {updated} mis à jour")

if __name__ == '__main__':
    print("=" * 50)
    print("  CommSight — Seed des plans d'abonnement")
    print("=" * 50)
    seed()