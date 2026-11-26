"""Les catégories de réclamations, et l'équipe de la banque qui traite chacune.

Le régulateur américain (CFPB) a renommé plusieurs produits au fil des années : un même produit
peut apparaître sous plusieurs noms. On les ramène à 9 catégories stables, une par équipe.
"""

LABELS: tuple[str, ...] = (
    "credit_reporting",
    "debt_collection",
    "cards",
    "bank_account",
    "mortgage",
    "money_transfer",
    "personal_loan",
    "vehicle_loan",
    "student_loan",
)

TEAMS_FR: dict[str, str] = {
    "credit_reporting": "Rapports de crédit",
    "debt_collection": "Recouvrement",
    "cards": "Cartes bancaires",
    "bank_account": "Comptes courants et épargne",
    "mortgage": "Prêts immobiliers",
    "money_transfer": "Virements et paiements",
    "personal_loan": "Prêts personnels",
    "vehicle_loan": "Crédits auto",
    "student_loan": "Prêts étudiants",
}

# Descriptions données au LLM de comparaison (en anglais, comme les réclamations)
DESCRIPTIONS_EN: dict[str, str] = {
    "credit_reporting": "credit reports, credit scores, credit bureaus, disputes on a report",
    "debt_collection": "debt collectors, collection calls or letters, debts not owed",
    "cards": "credit cards and prepaid cards: billing, fees, rewards, card fraud, card disputes",
    "bank_account": "checking or savings accounts: deposits, withdrawals, overdrafts, closure",
    "mortgage": "home loans: mortgage payments, escrow, servicing, modification, foreclosure",
    "money_transfer": "money transfers, payment apps, wire transfers, virtual currency",
    "personal_loan": "payday loans, title loans, personal loans, installment or advance loans",
    "vehicle_loan": "car loans and vehicle leases: payments, repossession, lease terms",
    "student_loan": "federal or private student loans: servicing, repayment plans, forgiveness",
}

PRODUCT_TO_LABEL: dict[str, str] = {
    "Credit reporting, credit repair services, or other personal consumer reports": (
        "credit_reporting"
    ),
    "Credit reporting or other personal consumer reports": "credit_reporting",
    "Credit reporting": "credit_reporting",
    "Debt collection": "debt_collection",
    "Credit card or prepaid card": "cards",
    "Credit card": "cards",
    "Prepaid card": "cards",
    "Checking or savings account": "bank_account",
    "Bank account or service": "bank_account",
    "Mortgage": "mortgage",
    "Money transfer, virtual currency, or money service": "money_transfer",
    "Money transfers": "money_transfer",
    "Virtual currency": "money_transfer",
    "Payday loan, title loan, or personal loan": "personal_loan",
    "Payday loan, title loan, personal loan, or advance loan": "personal_loan",
    "Payday loan": "personal_loan",
    "Vehicle loan or lease": "vehicle_loan",
    "Student loan": "student_loan",
}

# Produits volontairement écartés, et pourquoi
EXCLUDED_PRODUCTS: dict[str, str] = {
    "Debt or credit management": "catégorie créée en 2023, en cours de période, et très petite",
    "Consumer Loan": "ancienne catégorie qui mélangeait crédits auto et prêts personnels",
    "Other financial service": "catégorie fourre-tout, sans équipe dédiée",
}


def label_for_product(product: str | None) -> str | None:
    """Catégorie d'un produit du CFPB, ou None si le produit est écarté ou inconnu."""
    if product is None:
        return None
    return PRODUCT_TO_LABEL.get(product.strip())
