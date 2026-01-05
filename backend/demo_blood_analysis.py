"""
Demo: Blood Report Analysis
Shows exactly what the system can do with blood test reports
"""

from utils.blood_report_analyzer import BloodReportAnalyzer

# Initialize analyzer
analyzer = BloodReportAnalyzer()

# Example blood report text (as extracted by OCR)
sample_report = """
PATIENT: John Doe
AGE: 45 Years
GENDER: Male
DATE: January 6, 2026

COMPLETE BLOOD COUNT (CBC)
----------------------------
Hemoglobin: 11.2 g/dL
RBC Count: 4.2 million/µL
WBC Count: 12500 /µL
Platelet Count: 180000 /µL
Hematocrit: 38 %

LIPID PROFILE
----------------------------
Total Cholesterol: 245 mg/dL
LDL Cholesterol: 145 mg/dL
HDL Cholesterol: 38 mg/dL
Triglycerides: 210 mg/dL

BLOOD GLUCOSE
----------------------------
Fasting Blood Sugar: 125 mg/dL
HbA1c: 6.2 %

LIVER FUNCTION TEST
----------------------------
ALT (SGPT): 62 U/L
AST (SGOT): 48 U/L
Total Bilirubin: 1.0 mg/dL
Albumin: 4.2 g/dL

KIDNEY FUNCTION TEST
----------------------------
Creatinine: 1.1 mg/dL
BUN: 18 mg/dL
Uric Acid: 7.8 mg/dL
"""

print("\n" + "="*70)
print("MEDICAL REPORT AI - BLOOD REPORT ANALYSIS DEMO")
print("="*70)

print("\n📄 SAMPLE BLOOD REPORT (After OCR Extraction):")
print(sample_report)

print("\n🔍 ANALYZING REPORT...")
print("-"*70)

# Analyze the report
analysis = analyzer.analyze_report(sample_report, gender='male')

# Generate human-readable report
report_text = analyzer.generate_report_text(analysis)

print(report_text)

print("\n" + "="*70)
print("📊 DETAILED ANALYSIS (JSON Format for Frontend):")
print("="*70)

import json
print(json.dumps(analysis, indent=2))

print("\n" + "="*70)
print("💡 WHAT THIS DEMO SHOWS:")
print("="*70)
print("""
1. ✅ System extracts text from your uploaded blood report (using OCR)
2. ✅ Identifies all blood test values (Hemoglobin, WBC, Cholesterol, etc.)
3. ✅ Compares each value against normal reference ranges
4. ✅ Determines if values are LOW, NORMAL, or HIGH
5. ✅ Provides specific recommendations for abnormal values
6. ✅ Generates an overall health summary

This works with REAL BLOOD TEST REPORTS from labs!
""")

print("\n" + "="*70)
print("🎯 HOW TO USE:")
print("="*70)
print("""
1. Upload your blood report (PDF/Image) to the system
2. System uses OCR to extract the text
3. System automatically identifies it's a blood report
4. You get instant analysis with:
   - Which values are normal/abnormal
   - What each abnormal value means
   - Specific recommendations for your health
   - Overall health summary

You can also ask questions like:
- "What does high cholesterol mean?"
- "Why is my hemoglobin low?"
- "What should I eat to improve my results?"
""")

print("\n✨ DEMO COMPLETE!")
