import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

N_CLIENTS = 500

INDUSTRIES = ["Manufacturing", "Real Estate", "Shipping", "Tech", "Energy", "Retail"]


def generate_ci_clients(n=N_CLIENTS):
    records = []

    for i in range(n):
        industry = random.choice(INDUSTRIES)
        months_as_client = np.random.randint(6, 180)
        num_products = np.random.randint(1, 6)
        fx_volume = (
            np.random.exponential(500_000)
            if industry in ["Shipping", "Energy"]
            else np.random.exponential(100_000)
        )
        lending_util = np.random.beta(2, 5)
        last_rm_contact = np.random.randint(1, 180)
        annual_revenue = np.random.lognormal(mean=14, sigma=1.5)

        # Churn probability: less RM contact, fewer products, low utilization = higher churn
        churn_prob = (
            0.05
            + (last_rm_contact / 180) * 0.3
            + (1 - lending_util) * 0.2
            - (num_products / 6) * 0.25
            - (months_as_client / 180) * 0.1
        )
        churn_prob = np.clip(churn_prob, 0.02, 0.95)
        is_churned = int(np.random.random() < churn_prob)

        # Cross-sell potential: high revenue, long relationship, few products = high potential
        cross_sell = (
            np.log(annual_revenue) / 20
            + (months_as_client / 180) * 0.3
            - (num_products / 6) * 0.4
            + np.random.normal(0, 0.05)
        )
        cross_sell = np.clip(cross_sell, 0, 1)

        records.append({
            "client_id": f"CI_{i:04d}",
            "industry": industry,
            "annual_revenue_sek": round(annual_revenue),
            "months_as_client": months_as_client,
            "num_products_held": num_products,
            "fx_volume_3m_sek": round(fx_volume),
            "lending_utilization_pct": round(lending_util, 3),
            "last_rm_contact_days": last_rm_contact,
            "is_churned": is_churned,
            "cross_sell_score": round(cross_sell, 3),
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_ci_clients()
    df.to_csv("ci_clients.csv", index=False)
    print(f"Done! Generated {len(df)} client records.")
    print(f"Churn rate: {df['is_churned'].mean():.1%}")
    print(f"\nPreview:")
    print(df.head())
    