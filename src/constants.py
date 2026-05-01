"""
src/constants.py
----------------
Central configuration file for the DailyMed NLP pipeline.
Standardizes LOINC codes, groups them into semantic Super-Groups, 
and defines token-based chunking constraints.
"""

# --- MASTER LOINC MAPPING ---
# Maps specific medical codes to clean, human-readable categories.
MASTER_LOINC_MAP = {
    # Core Clinical Safety & Usage
    "34066-1": "BOXED_WARNING",
    "34067-9": "INDICATIONS_AND_USAGE",
    "34068-7": "DOSAGE_AND_ADMINISTRATION",
    "34070-3": "CONTRAINDICATIONS",
    "34071-1": "WARNINGS",
    "43685-7": "WARNINGS_AND_PRECAUTIONS",
    "34084-4": "ADVERSE_REACTIONS",
    "34088-5": "OVERDOSAGE",
    "42232-9": "PRECAUTIONS",

    # Specific Populations
    "42228-7": "PREGNANCY",
    "34077-8": "PREGNANCY_TERATOGENIC",
    "34078-6": "PREGNANCY_NON_TERATOGENIC",
    "34079-4": "LABOR_AND_DELIVERY",
    "34080-2": "NURSING_MOTHERS",
    "77290-5": "LACTATION",
    "34081-0": "PEDIATRIC_USE",
    "34082-8": "GERIATRIC_USE",
    "77291-3": "REPRODUCTIVE_POTENTIAL",
    "43684-0": "SPECIFIC_POPULATIONS",
    "88828-9": "RENAL_IMPAIRMENT",
    "88829-7": "HEPATIC_IMPAIRMENT",

    # Clinical Science & Pharmacology
    "34073-7": "DRUG_INTERACTIONS",
    "34074-5": "LAB_TEST_INTERACTIONS",
    "34075-2": "LABORATORY_TESTS",
    "34090-1": "CLINICAL_PHARMACOLOGY",
    "43679-0": "MECHANISM_OF_ACTION",
    "43681-6": "PHARMACODYNAMICS",
    "43682-4": "PHARMACOKINETICS",
    "34092-7": "CLINICAL_STUDIES",
    "90374-0": "CLINICAL_TRIALS_EXPERIENCE",
    "90375-7": "POSTMARKETING_EXPERIENCE",
    "49489-8": "MICROBIOLOGY",
    "34083-6": "NONCLINICAL_TOXICOLOGY",
    "34091-9": "ANIMAL_TOXICOLOGY",
    "66106-6": "PHARMACOGENOMICS",
    "88830-5": "IMMUNOGENICITY",

    # Product Logistics & Ingredients
    "34089-3": "DESCRIPTION",
    "43678-2": "DOSAGE_FORMS_STRENGTHS",
    "34069-5": "HOW_SUPPLIED_STORAGE",
    "44425-7": "STORAGE_HANDLING",
    "51727-6": "INACTIVE_INGREDIENTS",
    "43683-2": "RECENT_MAJOR_CHANGES",
    "60558-4": "HANDLING_AND_STERILIZATION",
    "48780-1": "PRODUCT_DATA_LISTING",
    "51945-4": "PACKAGE_LABEL_TEXT",
    "48779-3": "TABLE_OF_CONTENTS",

    # Patient-Facing
    "34076-0": "PATIENT_COUNSELING_INFO",
    "42231-1": "MEDGUIDE_SECTION",
    "42230-3": "PATIENT_PACKAGE_INSERT",
    "68498-5": "PATIENT_MED_INFO",
    "38056-8": "SUPPLEMENTAL_PATIENT_MATERIAL",
    "82598-4": "REMS_MEDICATION_GUIDE",
    "88436-1": "PATIENT_COUNSELING_TEXT",

    # OTC & Abuse
    "42227-9": "DRUG_ABUSE_DEPENDENCE",
    "34085-1": "CONTROLLED_SUBSTANCE",
    "34086-9": "ABUSE",
    "34087-7": "DEPENDENCE",
    "55106-9": "OTC_ACTIVE_INGREDIENT",
    "55105-1": "OTC_PURPOSE",

    # Catch-All
    "42229-5": "UNCLASSIFIED",
    "None": "OTHER"
}

# --- SUPER GROUP MAPPING ---
# Consolidates the above categories into 6 Semantic Super-Groups.
# This is used by the Dynamic Accumulator to group related sections.
SUPER_GROUP_MAP = {
    # Group: SAFETY_AND_RISK
    "BOXED_WARNING": "SAFETY_RISK",
    "WARNINGS": "SAFETY_RISK",
    "WARNINGS_AND_PRECAUTIONS": "SAFETY_RISK",
    "CONTRAINDICATIONS": "SAFETY_RISK",
    "PRECAUTIONS": "SAFETY_RISK",
    "DRUG_ABUSE_DEPENDENCE": "SAFETY_RISK",
    "CONTROLLED_SUBSTANCE": "SAFETY_RISK",
    "ABUSE": "SAFETY_RISK",
    "DEPENDENCE": "SAFETY_RISK",

    # Group: CLINICAL_USAGE
    "INDICATIONS_AND_USAGE": "USAGE_CLINICAL",
    "DOSAGE_AND_ADMINISTRATION": "USAGE_CLINICAL",
    "DOSAGE_FORMS_STRENGTHS": "USAGE_CLINICAL",
    "OTC_ACTIVE_INGREDIENT": "USAGE_CLINICAL",
    "OTC_PURPOSE": "USAGE_CLINICAL",

    # Group: ADVERSE_EVENTS
    "ADVERSE_REACTIONS": "ADVERSE_INTERACTIONS",
    "DRUG_INTERACTIONS": "ADVERSE_INTERACTIONS",
    "LAB_TEST_INTERACTIONS": "ADVERSE_INTERACTIONS",
    "OVERDOSAGE": "ADVERSE_INTERACTIONS",
    "POSTMARKETING_EXPERIENCE": "ADVERSE_INTERACTIONS",
    "LABORATORY_TESTS": "ADVERSE_INTERACTIONS",

    # Group: SPECIAL_POPULATIONS
    "PREGNANCY": "SPECIAL_POPULATIONS",
    "PREGNANCY_TERATOGENIC": "SPECIAL_POPULATIONS",
    "PREGNANCY_NON_TERATOGENIC": "SPECIAL_POPULATIONS",
    "NURSING_MOTHERS": "SPECIAL_POPULATIONS",
    "PEDIATRIC_USE": "SPECIAL_POPULATIONS",
    "GERIATRIC_USE": "SPECIAL_POPULATIONS",
    "LACTATION": "SPECIAL_POPULATIONS",
    "SPECIFIC_POPULATIONS": "SPECIAL_POPULATIONS",
    "LABOR_AND_DELIVERY": "SPECIAL_POPULATIONS",
    "REPRODUCTIVE_POTENTIAL": "SPECIAL_POPULATIONS",
    "RENAL_IMPAIRMENT": "SPECIAL_POPULATIONS",
    "HEPATIC_IMPAIRMENT": "SPECIAL_POPULATIONS",

    # Group: PHARMACOLOGY_SCIENCE
    "CLINICAL_PHARMACOLOGY": "PHARMACOLOGY",
    "MECHANISM_OF_ACTION": "PHARMACOLOGY",
    "PHARMACODYNAMICS": "PHARMACOLOGY",
    "PHARMACOKINETICS": "PHARMACOLOGY",
    "CLINICAL_STUDIES": "PHARMACOLOGY",
    "CLINICAL_TRIALS_EXPERIENCE": "PHARMACOLOGY",
    "NONCLINICAL_TOXICOLOGY": "PHARMACOLOGY",
    "ANIMAL_TOXICOLOGY": "PHARMACOLOGY",
    "MICROBIOLOGY": "PHARMACOLOGY",
    "PHARMACOGENOMICS": "PHARMACOLOGY",
    "IMMUNOGENICITY": "PHARMACOLOGY",

    # Group: PRODUCT_LOGISTICS
    "DESCRIPTION": "PRODUCT_LOGISTICS",
    "HOW_SUPPLIED_STORAGE": "PRODUCT_LOGISTICS",
    "STORAGE_HANDLING": "PRODUCT_LOGISTICS",
    "INACTIVE_INGREDIENTS": "PRODUCT_LOGISTICS",
    "PRODUCT_DATA_LISTING": "PRODUCT_LOGISTICS",
    "PACKAGE_LABEL_TEXT": "PRODUCT_LOGISTICS",
    "RECENT_MAJOR_CHANGES": "PRODUCT_LOGISTICS"
}

# --- SEARCH & CHUNKING SETTINGS ---
# Optimized for all-mpnet-base-v2 (512 token context window)

# Use 480 instead of 512 to leave space for the metadata headers we inject
MAX_TOKEN_LENGTH = 480  
CHUNK_OVERLAP_TOKENS = 50 

# High-priority groups to be analyzed first in the chatbot logic
HIGH_PRIORITY_GROUPS = [
    "SAFETY_RISK", 
    "USAGE_CLINICAL", 
    "ADVERSE_INTERACTIONS"
]
# # """
# # src/constants.py
# # ----------------
# # Central configuration file for the DailyMed NLP pipeline.
# # Implements a "Clinical Grouping" strategy to reduce chunk noise and 
# # increase semantic context for the RAG system.
# # """

# # # --- MASTER CLINICAL GROUPS ---
# # # We map 90+ granular LOINC codes into 9 Master Categories.
# # # This ensures that related information (e.g., all pregnancy info) stays together.

# # MASTER_LOINC_MAP = {
# #     # 1. GROUP: SAFETY_WARNINGS
# #     "34066-1": "SAFETY_WARNINGS",            # Boxed Warning
# #     "34071-1": "SAFETY_WARNINGS",            # Warnings
# #     "43685-7": "SAFETY_WARNINGS",            # Warnings and Precautions
# #     "34070-3": "SAFETY_WARNINGS",            # Contraindications
# #     "42232-9": "SAFETY_WARNINGS",            # Precautions
# #     "50741-8": "SAFETY_WARNINGS",            # Safe Handling Warning
# #     "54433-8": "SAFETY_WARNINGS",            # User Safety Warnings

# #     # 2. GROUP: USAGE_AND_DOSAGE
# #     "34067-9": "USAGE_AND_DOSAGE",           # Indications & Usage
# #     "34068-7": "USAGE_AND_DOSAGE",           # Dosage & Administration
# #     "50745-9": "USAGE_AND_DOSAGE",           # Veterinary Indications
# #     "59845-8": "USAGE_AND_DOSAGE",           # Instructions for Use
# #     "43678-2": "USAGE_AND_DOSAGE",           # Dosage Forms & Strengths
# #     "60560-0": "USAGE_AND_DOSAGE",           # Intended Use (Device)

# #     # 3. GROUP: SPECIAL_POPULATIONS
# #     "42228-7": "SPECIAL_POPULATIONS",        # Pregnancy
# #     "34077-8": "SPECIAL_POPULATIONS",        # Teratogenic Effects
# #     "34078-6": "SPECIAL_POPULATIONS",        # Non-Teratogenic Effects
# #     "34079-4": "SPECIAL_POPULATIONS",        # Labor and Delivery
# #     "34080-2": "SPECIAL_POPULATIONS",        # Nursing Mothers
# #     "77290-5": "SPECIAL_POPULATIONS",        # Lactation
# #     "34081-0": "SPECIAL_POPULATIONS",        # Pediatric Use
# #     "34082-8": "SPECIAL_POPULATIONS",        # Geriatric Use
# #     "77291-3": "SPECIAL_POPULATIONS",        # Reproductive Potential
# #     "43684-0": "SPECIAL_POPULATIONS",        # Specific Populations
# #     "88828-9": "SPECIAL_POPULATIONS",        # Renal Impairment
# #     "88829-7": "SPECIAL_POPULATIONS",        # Hepatic Impairment

# #     # 4. GROUP: CLINICAL_SCIENCE
# #     "34090-1": "CLINICAL_SCIENCE",           # Clinical Pharmacology
# #     "43679-0": "CLINICAL_SCIENCE",           # Mechanism of Action
# #     "43681-6": "CLINICAL_SCIENCE",           # Pharmacodynamics
# #     "43682-4": "CLINICAL_SCIENCE",           # Pharmacokinetics
# #     "34092-7": "CLINICAL_SCIENCE",           # Clinical Studies
# #     "90374-0": "CLINICAL_SCIENCE",           # Clinical Trials Experience
# #     "49489-8": "CLINICAL_SCIENCE",           # Microbiology
# #     "34083-6": "CLINICAL_SCIENCE",           # Nonclinical Toxicology
# #     "34091-9": "CLINICAL_SCIENCE",           # Animal Toxicology
# #     "66106-6": "CLINICAL_SCIENCE",           # Pharmacogenomics
# #     "88830-5": "CLINICAL_SCIENCE",           # Immunogenicity

# #     # 5. GROUP: REACTIONS_AND_INTERACTIONS
# #     "34084-4": "REACTIONS_INTERACTIONS",      # Adverse Reactions
# #     "34073-7": "REACTIONS_INTERACTIONS",      # Drug Interactions
# #     "34074-5": "REACTIONS_INTERACTIONS",      # Lab Test Interactions
# #     "34075-2": "REACTIONS_INTERACTIONS",      # Laboratory Tests
# #     "90375-7": "REACTIONS_INTERACTIONS",      # Postmarketing Experience
# #     "34088-5": "REACTIONS_INTERACTIONS",      # Overdosage

# #     # 6. GROUP: PRODUCT_LOGISTICS
# #     "34089-3": "PRODUCT_LOGISTICS",          # Description
# #     "34069-5": "PRODUCT_LOGISTICS",          # How Supplied
# #     "44425-7": "PRODUCT_LOGISTICS",          # Storage and Handling
# #     "51727-6": "PRODUCT_LOGISTICS",          # Inactive Ingredients
# #     "48780-1": "PRODUCT_LOGISTICS",          # SPL Product Data (Ingredients dump)
# #     "51945-4": "PRODUCT_LOGISTICS",          # Package Label Text
# #     "43683-2": "PRODUCT_LOGISTICS",          # Recent Major Changes
# #     "60558-4": "PRODUCT_LOGISTICS",          # Handling/Sterilization
# #     "69763-1": "PRODUCT_LOGISTICS",          # Disposal/Waste

# #     # 7. GROUP: PATIENT_INFORMATION
# #     "34076-0": "PATIENT_INFO",               # Patient Counseling Info
# #     "42231-1": "PATIENT_INFO",               # Medguide
# #     "42230-3": "PATIENT_INFO",               # Patient Package Insert
# #     "68498-5": "PATIENT_INFO",               # Patient Med Info
# #     "82598-4": "PATIENT_INFO",               # REMS Medguide
# #     "88436-1": "PATIENT_INFO",               # Patient Counseling Text
# #     "38056-8": "PATIENT_INFO",               # Supplemental Material

# #     # 8. GROUP: OTC_SPECIFIC
# #     "55106-9": "OTC_DATA",                   # OTC Active Ingredient
# #     "55105-1": "OTC_DATA",                   # OTC Purpose
# #     "50570-1": "OTC_DATA",                   # OTC Do Not Use
# #     "50569-3": "OTC_DATA",                   # OTC Ask Doctor
# #     "50568-5": "OTC_DATA",                   # OTC Ask Pharmacist
# #     "50567-7": "OTC_DATA",                   # OTC When Using
# #     "50566-9": "OTC_DATA",                   # OTC Stop Use
# #     "50565-1": "OTC_DATA",                   # OTC Keep Out of Reach
# #     "53414-9": "OTC_DATA",                   # OTC Pregnancy/Breastfeeding
# #     "53413-1": "OTC_DATA",                   # OTC Questions

# #     # 9. GROUP: ADMINISTRATIVE
# #     "48779-3": "ADMIN_STRUCTURE",            # Table of Contents
# #     "69758-1": "ADMIN_STRUCTURE",            # Diagram Labels
# #     "69718-5": "ADMIN_STRUCTURE",            # Identity Statement
# #     "69719-3": "ADMIN_STRUCTURE",            # Health Claim
# #     "60559-2": "ADMIN_STRUCTURE",            # Device Components
# #     "60563-4": "ADMIN_STRUCTURE",            # Device Safety Summary
# #     "69759-9": "ADMIN_STRUCTURE",            # Device Risks
# #     "69760-7": "ADMIN_STRUCTURE",            # Compatible Accessories
# #     "69761-5": "ADMIN_STRUCTURE",            # Alarms
# #     "71744-7": "ADMIN_STRUCTURE",            # HCP Letter
# #     "34093-5": "ADMIN_STRUCTURE",            # References

# #     # Catch-All
# #     "42229-5": "UNCLASSIFIED",
# #     "None": "OTHER"
# # }

# # # --- PRIORITY SETTINGS ---
# # # Use these categories to boost relevance in retrieval
# # HIGH_PRIORITY_GROUPS = [
# #     "SAFETY_WARNINGS",
# #     "USAGE_AND_DOSAGE",
# #     "REACTIONS_INTERACTIONS"
# # ]

# # # --- CHUNKING CONFIGURATION ---
# # CHUNK_SIZE = 1500   # Slightly larger since we are grouping sections
# # CHUNK_OVERLAP = 150

# """
# src/constants.py
# ----------------
# Central configuration file for the DailyMed NLP pipeline.
# Standardizes every discovered LOINC code from the 15,094 file audit.
# This version implements a "No Data Loss" policy—all sections are categorized.
# """

# # --- MASTER LOINC MAPPING ---
# # Every code discovered in the audit is mapped here to a clean snake_case category.
# MASTER_LOINC_MAP = {
#     # Core Clinical Safety & Usage
#     "34066-1": "BOXED_WARNING",
#     "34067-9": "INDICATIONS_AND_USAGE",
#     "34068-7": "DOSAGE_AND_ADMINISTRATION",
#     "34070-3": "CONTRAINDICATIONS",
#     "34071-1": "WARNINGS",
#     "43685-7": "WARNINGS_AND_PRECAUTIONS",
#     "34084-4": "ADVERSE_REACTIONS",
#     "34088-5": "OVERDOSAGE",
#     "42232-9": "PRECAUTIONS",

#     # Specific Populations
#     "42228-7": "PREGNANCY",
#     "34077-8": "PREGNANCY_TERATOGENIC",
#     "34078-6": "PREGNANCY_NON_TERATOGENIC",
#     "34079-4": "LABOR_AND_DELIVERY",
#     "34080-2": "NURSING_MOTHERS",
#     "77290-5": "LACTATION",
#     "34081-0": "PEDIATRIC_USE",
#     "34082-8": "GERIATRIC_USE",
#     "77291-3": "REPRODUCTIVE_POTENTIAL",
#     "43684-0": "SPECIFIC_POPULATIONS",
#     "88828-9": "RENAL_IMPAIRMENT",
#     "88829-7": "HEPATIC_IMPAIRMENT",

#     # Clinical Science & Pharmacology
#     "34073-7": "DRUG_INTERACTIONS",
#     "34074-5": "LAB_TEST_INTERACTIONS",
#     "34075-2": "LABORATORY_TESTS",
#     "34090-1": "CLINICAL_PHARMACOLOGY",
#     "43679-0": "MECHANISM_OF_ACTION",
#     "43681-6": "PHARMACODYNAMICS",
#     "43682-4": "PHARMACOKINETICS",
#     "34092-7": "CLINICAL_STUDIES",
#     "90374-0": "CLINICAL_TRIALS_EXPERIENCE",
#     "90375-7": "POSTMARKETING_EXPERIENCE",
#     "49489-8": "MICROBIOLOGY",
#     "34083-6": "NONCLINICAL_TOXICOLOGY",
#     "34091-9": "ANIMAL_TOXICOLOGY",
#     "66106-6": "PHARMACOGENOMICS",
#     "88830-5": "IMMUNOGENICITY",

#     # Product Logistics, Chemistry & Ingredients
#     "34089-3": "DESCRIPTION",
#     "43678-2": "DOSAGE_FORMS_STRENGTHS",
#     "34069-5": "HOW_SUPPLIED_STORAGE",
#     "44425-7": "STORAGE_HANDLING",
#     "51727-6": "INACTIVE_INGREDIENTS",
#     "43683-2": "RECENT_MAJOR_CHANGES",
#     "60558-4": "HANDLING_AND_STERILIZATION",
#     "48780-1": "PRODUCT_DATA_LISTING",  # Keep: contains inactive ingredients
#     "51945-4": "PACKAGE_LABEL_TEXT",      # Keep: representation of box text
#     "48779-3": "TABLE_OF_CONTENTS",       # Keep: structural indexing

#     # Patient-Facing Materials
#     "34076-0": "PATIENT_COUNSELING_INFO",
#     "42231-1": "MEDGUIDE_SECTION",
#     "42230-3": "PATIENT_PACKAGE_INSERT",
#     "68498-5": "PATIENT_MED_INFO",
#     "38056-8": "SUPPLEMENTAL_PATIENT_MATERIAL",
#     "82598-4": "REMS_MEDICATION_GUIDE",
#     "88436-1": "PATIENT_COUNSELING_TEXT",

#     # Controlled Substances & Abuse
#     "42227-9": "DRUG_ABUSE_DEPENDENCE",
#     "34085-1": "CONTROLLED_SUBSTANCE",
#     "34086-9": "ABUSE",
#     "34087-7": "DEPENDENCE",

#     # OTC (Over-The-Counter) Facts
#     "55106-9": "OTC_ACTIVE_INGREDIENT",
#     "55105-1": "OTC_PURPOSE",
#     "50570-1": "OTC_DO_NOT_USE",
#     "50569-3": "OTC_ASK_DOCTOR",
#     "50568-5": "OTC_ASK_PHARMACIST",
#     "50567-7": "OTC_WHEN_USING",
#     "50566-9": "OTC_STOP_USE",
#     "50565-1": "OTC_KEEP_OUT_OF_REACH",
#     "53414-9": "OTC_PREGNANCY_BREASTFEEDING",
#     "53413-1": "OTC_QUESTIONS",
#     "59845-8": "INSTRUCTIONS_FOR_USE",

#     # Device & Healthcare Provider Info
#     "60560-0": "INTENDED_USE_DEVICE",
#     "60559-2": "DEVICE_COMPONENTS",
#     "60563-4": "DEVICE_SAFETY_SUMMARY",
#     "69763-1": "DISPOSAL_WASTE_HANDLING",
#     "69758-1": "DEVICE_DIAGRAM_LABELS",
#     "69718-5": "IDENTITY_STATEMENT",
#     "69719-3": "HEALTH_CLAIM",
#     "69759-9": "DEVICE_RISKS",
#     "69760-7": "COMPATIBLE_ACCESSORIES",
#     "69761-5": "DEVICE_ALARMS",
#     "71744-7": "HEALTHCARE_PROVIDER_LETTER",
#     "34093-5": "REFERENCES",

#     # Catch-All
#     "42229-5": "UNCLASSIFIED",
#     "None": "OTHER"
# }

# # --- PRIORITY SETTINGS ---
# # High-priority sections often cited in medical safety evaluations.
# HIGH_PRIORITY_SECTIONS = [
#     "BOXED_WARNING",
#     "CONTRAINDICATIONS",
#     "WARNINGS",
#     "INDICATIONS_AND_USAGE",
#     "DOSAGE_AND_ADMINISTRATION"
# ]

# # --- CHUNKING CONFIGURATION ---
# # Optimized for high-density medical text retrieval.
# CHUNK_SIZE = 1000  # Max characters per section chunk
# CHUNK_OVERLAP = 100 # Overlap for context retention