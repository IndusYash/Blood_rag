"""
Blood Report Analyzer
Analyzes text-based blood test reports and provides interpretations
"""

import re
from typing import Dict, List, Tuple, Any


class BloodReportAnalyzer:
    """Analyzes blood test reports and interprets values"""
    
    # Normal reference ranges (standard Indian/WHO guidelines)
    NORMAL_RANGES = {
        # Complete Blood Count (CBC)
        'hemoglobin': {
            'male': (13.0, 17.0, 'g/dL'),
            'female': (12.0, 15.0, 'g/dL'),
            'keywords': ['hemoglobin', 'hb', 'hgb']
        },
        'rbc': {
            'male': (4.5, 5.5, 'million/µL'),
            'female': (4.0, 5.0, 'million/µL'),
            'keywords': ['rbc', 'red blood cell', 'erythrocyte']
        },
        'wbc': {
            'normal': (4000, 11000, '/µL'),
            'keywords': ['wbc', 'white blood cell', 'leukocyte', 'total leucocyte']
        },
        'platelet': {
            'normal': (150000, 450000, '/µL'),
            'keywords': ['platelet', 'plt', 'thrombocyte']
        },
        'hematocrit': {
            'male': (40, 54, '%'),
            'female': (36, 46, '%'),
            'keywords': ['hematocrit', 'hct', 'pcv']
        },
        
        # Lipid Profile
        'cholesterol': {
            'normal': (0, 200, 'mg/dL'),
            'keywords': ['total cholesterol', 'cholesterol']
        },
        'ldl': {
            'normal': (0, 100, 'mg/dL'),
            'keywords': ['ldl', 'low density lipoprotein']
        },
        'hdl': {
            'male': (40, 999, 'mg/dL'),
            'female': (50, 999, 'mg/dL'),
            'keywords': ['hdl', 'high density lipoprotein']
        },
        'triglycerides': {
            'normal': (0, 150, 'mg/dL'),
            'keywords': ['triglyceride', 'tg']
        },
        
        # Blood Glucose
        'fasting_glucose': {
            'normal': (70, 100, 'mg/dL'),
            'keywords': ['fasting glucose', 'fasting blood sugar', 'fbs']
        },
        'random_glucose': {
            'normal': (70, 140, 'mg/dL'),
            'keywords': ['random glucose', 'random blood sugar', 'rbs']
        },
        'hba1c': {
            'normal': (4.0, 5.6, '%'),
            'keywords': ['hba1c', 'glycated hemoglobin', 'glycosylated hemoglobin']
        },
        
        # Liver Function
        'alt': {
            'normal': (7, 56, 'U/L'),
            'keywords': ['alt', 'sgpt', 'alanine aminotransferase']
        },
        'ast': {
            'normal': (10, 40, 'U/L'),
            'keywords': ['ast', 'sgot', 'aspartate aminotransferase']
        },
        'bilirubin': {
            'normal': (0.1, 1.2, 'mg/dL'),
            'keywords': ['total bilirubin', 'bilirubin']
        },
        'albumin': {
            'normal': (3.5, 5.5, 'g/dL'),
            'keywords': ['albumin', 'serum albumin']
        },
        
        # Kidney Function
        'creatinine': {
            'male': (0.7, 1.3, 'mg/dL'),
            'female': (0.6, 1.1, 'mg/dL'),
            'keywords': ['creatinine', 'serum creatinine']
        },
        'bun': {
            'normal': (7, 20, 'mg/dL'),
            'keywords': ['bun', 'blood urea nitrogen', 'urea']
        },
        'uric_acid': {
            'male': (3.4, 7.0, 'mg/dL'),
            'female': (2.4, 6.0, 'mg/dL'),
            'keywords': ['uric acid', 'urate']
        },
        
        # Thyroid
        'tsh': {
            'normal': (0.4, 4.0, 'mIU/L'),
            'keywords': ['tsh', 'thyroid stimulating hormone']
        },
        't3': {
            'normal': (80, 200, 'ng/dL'),
            'keywords': ['t3', 'triiodothyronine']
        },
        't4': {
            'normal': (4.5, 12.0, 'µg/dL'),
            'keywords': ['t4', 'thyroxine']
        }
    }
    
    def __init__(self):
        """Initialize analyzer"""
        pass
    
    def extract_values(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract test values from report text
        
        Args:
            text: Extracted text from blood report
            
        Returns:
            List of dictionaries with test names, values, and units
        """
        results = []
        
        # Common patterns for test results
        # Pattern 1: "Test Name: Value Unit"
        # Pattern 2: "Test Name Value Unit"
        # Pattern 3: "Test Name: Value"
        
        lines = text.split('\n')
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
            
            # Try to extract value patterns
            # Pattern: number with optional decimal and unit
            value_pattern = r'(\d+\.?\d*)\s*([a-zA-Z/%µ]+)?'
            
            for test_name, test_info in self.NORMAL_RANGES.items():
                keywords = test_info['keywords']
                
                for keyword in keywords:
                    # Case-insensitive search
                    if keyword.lower() in line.lower():
                        # Try to extract the value
                        match = re.search(value_pattern, line)
                        if match:
                            value = float(match.group(1))
                            unit = match.group(2) if match.group(2) else ''
                            
                            results.append({
                                'test_name': test_name,
                                'display_name': keyword.title(),
                                'value': value,
                                'unit': unit,
                                'line': line.strip()
                            })
                            break
        
        return results
    
    def interpret_value(self, test_name: str, value: float, gender: str = 'normal') -> Dict[str, Any]:
        """
        Interpret a test value
        
        Args:
            test_name: Name of the test
            value: Test value
            gender: 'male', 'female', or 'normal'
            
        Returns:
            Dictionary with interpretation
        """
        if test_name not in self.NORMAL_RANGES:
            return {
                'status': 'unknown',
                'message': 'Test not in reference database'
            }
        
        test_info = self.NORMAL_RANGES[test_name]
        
        # Get appropriate range
        if gender in test_info:
            min_val, max_val, unit = test_info[gender]
        elif 'normal' in test_info:
            min_val, max_val, unit = test_info['normal']
        else:
            min_val, max_val, unit = test_info['male']  # Default
        
        # Determine status
        if value < min_val:
            status = 'low'
            severity = 'mild' if value >= min_val * 0.8 else 'significant'
            message = f'Below normal range ({min_val}-{max_val} {unit})'
        elif value > max_val:
            status = 'high'
            severity = 'mild' if value <= max_val * 1.2 else 'significant'
            message = f'Above normal range ({min_val}-{max_val} {unit})'
        else:
            status = 'normal'
            severity = 'none'
            message = f'Within normal range ({min_val}-{max_val} {unit})'
        
        return {
            'status': status,
            'severity': severity,
            'message': message,
            'normal_range': f'{min_val}-{max_val} {unit}',
            'deviation': round(((value - ((min_val + max_val) / 2)) / ((max_val - min_val) / 2)) * 100, 1)
        }
    
    def get_recommendations(self, test_name: str, status: str) -> List[str]:
        """
        Get recommendations based on test result
        
        Args:
            test_name: Name of the test
            status: 'low', 'high', or 'normal'
            
        Returns:
            List of recommendations
        """
        recommendations = {
            'hemoglobin': {
                'low': [
                    'Increase iron-rich foods (spinach, lentils, red meat)',
                    'Consider iron supplementation after consulting doctor',
                    'Check for internal bleeding or chronic diseases',
                    'Ensure adequate vitamin B12 and folic acid intake'
                ],
                'high': [
                    'Stay well hydrated',
                    'Consult doctor to rule out polycythemia',
                    'Monitor oxygen levels if living at high altitude'
                ]
            },
            'wbc': {
                'low': [
                    'Boost immune system with vitamin C',
                    'Rule out viral infections or bone marrow issues',
                    'Consult doctor for possible medication side effects'
                ],
                'high': [
                    'Check for infections or inflammation',
                    'Rule out leukemia or immune disorders',
                    'Consult doctor immediately if severely elevated'
                ]
            },
            'cholesterol': {
                'high': [
                    'Reduce saturated fat intake',
                    'Increase fiber-rich foods',
                    'Regular exercise (30 min daily)',
                    'Consider statins if prescribed by doctor'
                ]
            },
            'fasting_glucose': {
                'high': [
                    'Reduce refined carbohydrates and sugar',
                    'Regular physical activity',
                    'Monitor fasting glucose regularly',
                    'Consult doctor for diabetes screening (HbA1c test)'
                ],
                'low': [
                    'Eat regular meals, avoid skipping',
                    'Check for reactive hypoglycemia',
                    'Rule out insulin-producing tumors (rare)'
                ]
            },
            'creatinine': {
                'high': [
                    'Stay well hydrated',
                    'Reduce protein intake if excessive',
                    'Consult doctor to assess kidney function (eGFR)',
                    'Monitor blood pressure'
                ]
            },
            'tsh': {
                'high': [
                    'May indicate hypothyroidism',
                    'Consult endocrinologist',
                    'May need thyroid hormone replacement'
                ],
                'low': [
                    'May indicate hyperthyroidism',
                    'Consult endocrinologist',
                    'Check T3 and T4 levels'
                ]
            }
        }
        
        if test_name in recommendations and status in recommendations[test_name]:
            return recommendations[test_name][status]
        
        # Default recommendations
        if status == 'high' or status == 'low':
            return [
                'Consult your healthcare provider for detailed interpretation',
                'Repeat test to confirm results',
                'Discuss symptoms and medical history with doctor'
            ]
        
        return ['Values are within normal range. Maintain healthy lifestyle.']
    
    def analyze_report(self, text: str, gender: str = 'normal') -> Dict[str, Any]:
        """
        Complete analysis of blood report
        
        Args:
            text: Extracted text from report
            gender: Patient gender ('male', 'female', or 'normal')
            
        Returns:
            Complete analysis with interpretations and recommendations
        """
        # Extract values
        extracted = self.extract_values(text)
        
        if not extracted:
            return {
                'success': False,
                'message': 'No recognizable blood test values found in the text',
                'suggestion': 'Make sure the image is clear and contains standard blood test results'
            }
        
        # Analyze each value
        results = []
        abnormal_count = 0
        
        for item in extracted:
            interpretation = self.interpret_value(
                item['test_name'],
                item['value'],
                gender
            )
            
            if interpretation['status'] != 'normal':
                abnormal_count += 1
            
            recommendations = self.get_recommendations(
                item['test_name'],
                interpretation['status']
            )
            
            results.append({
                **item,
                'interpretation': interpretation,
                'recommendations': recommendations
            })
        
        # Overall summary
        if abnormal_count == 0:
            overall_status = 'healthy'
            summary = 'All detected values are within normal ranges. Keep up the good work!'
        elif abnormal_count <= 2:
            overall_status = 'minor_concerns'
            summary = f'{abnormal_count} value(s) outside normal range. Consult your doctor for guidance.'
        else:
            overall_status = 'needs_attention'
            summary = f'{abnormal_count} values outside normal range. Please consult your healthcare provider soon.'
        
        return {
            'success': True,
            'overall_status': overall_status,
            'summary': summary,
            'total_tests': len(results),
            'abnormal_count': abnormal_count,
            'results': results
        }
    
    def generate_report_text(self, analysis: Dict[str, Any]) -> str:
        """
        Generate human-readable report text
        
        Args:
            analysis: Analysis results
            
        Returns:
            Formatted report text
        """
        if not analysis.get('success'):
            return analysis.get('message', 'Analysis failed')
        
        report = []
        report.append("="*60)
        report.append("BLOOD REPORT ANALYSIS")
        report.append("="*60)
        report.append(f"\n{analysis['summary']}\n")
        report.append(f"Tests Analyzed: {analysis['total_tests']}")
        report.append(f"Abnormal Results: {analysis['abnormal_count']}\n")
        
        for item in analysis['results']:
            report.append("-"*60)
            report.append(f"\n{item['display_name']}")
            report.append(f"  Value: {item['value']} {item['unit']}")
            report.append(f"  Status: {item['interpretation']['status'].upper()}")
            report.append(f"  Normal Range: {item['interpretation']['normal_range']}")
            report.append(f"  {item['interpretation']['message']}")
            
            if item['interpretation']['status'] != 'normal':
                report.append(f"\n  Recommendations:")
                for rec in item['recommendations']:
                    report.append(f"    • {rec}")
            report.append("")
        
        report.append("="*60)
        report.append("DISCLAIMER: This analysis is for informational purposes only.")
        report.append("Always consult a qualified healthcare provider for medical advice.")
        report.append("="*60)
        
        return '\n'.join(report)
