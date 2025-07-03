#!/usr/bin/env python
"""
Generate a PDF summary of safety audit reports for Shielded RecRL.

This script reads all safety report JSON files and creates a PDF summary.

Usage:
  python generate_pdf_summary.py
"""
import pathlib
import os
import json
import sys

def generate_summary_pdf():
    try:
        # Check for reportlab
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import Paragraph, Spacer
    except ImportError:
        print("Error: reportlab package not installed. Install with: pip install reportlab")
        return False

    root = pathlib.Path(os.getenv("PROJ", "."))
    reports_path = root / "docs"
    output_file = reports_path / "safety_summary.pdf"
    
    # Collect all report files
    report_files = list(reports_path.glob("safety_report_*.json"))
    if not report_files:
        print("Error: No safety report files found in", reports_path)
        return False
    
    # Load policy thresholds
    try:
        thresholds_file = reports_path / "policy_thresholds.yaml"
        if thresholds_file.exists():
            import yaml
            with open(thresholds_file) as f:
                thresholds = yaml.safe_load(f)
        else:
            thresholds = None
    except Exception as e:
        print(f"Warning: Could not load thresholds: {e}")
        thresholds = None
    
    # Create the PDF
    c = canvas.Canvas(str(output_file), pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Shielded RecRL: Safety Audit Summary")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Header line
    c.line(50, height - 80, width - 50, height - 80)
    
    # Policy thresholds section if available
    y_pos = height - 100
    if thresholds:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_pos, "Policy Thresholds:")
        y_pos -= 20
        c.setFont("Helvetica", 10)
        for key, value in thresholds.items():
            c.drawString(70, y_pos, f"{key}: {value}")
            y_pos -= 15
        y_pos -= 10
    
    # Report summaries
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_pos, "Safety Reports:")
    y_pos -= 20
    
    for report_file in sorted(report_files):
        try:
            with open(report_file) as f:
                report = json.load(f)
            
            # Extract dataset name from filename
            dataset = report_file.stem.replace('safety_report_', '')
            
            # Dataset header
            c.setFont("Helvetica-Bold", 11)
            c.drawString(50, y_pos, f"Dataset: {dataset}")
            y_pos -= 20
            
            # Toxicity metrics
            c.setFont("Helvetica", 10)
            if 'tox' in report:
                tox_mean = report['tox'].get('mean', 'N/A')
                tox_p95 = report['tox'].get('p95', 'N/A')
                c.drawString(70, y_pos, f"Toxicity: mean={tox_mean:.4f}, p95={tox_p95:.4f}")
                y_pos -= 15
            
            # Popularity bias
            if 'pop' in report:
                gini_base = report['pop'].get('gini_base', 'N/A')
                gini_new = report['pop'].get('gini_new', 'N/A')
                delta = report['pop'].get('delta', 'N/A')
                c.drawString(70, y_pos, f"Gini: base={gini_base:.4f}, new={gini_new:.4f}, delta={delta:.4f}")
                y_pos -= 15
            
            # Gender parity (ml25m only)
            if 'parity' in report and report['parity']:
                male = report['parity'].get('male_rate', 0)
                female = report['parity'].get('female_rate', 0)
                gap = abs(male - female)
                c.drawString(70, y_pos, f"Gender: M={male:.4f}, F={female:.4f}, gap={gap:.4f}")
                y_pos -= 15
            
            # Privacy
            if 'privacy' in report:
                priv = report['privacy']
                c.drawString(70, y_pos, f"Privacy leakage rate: {priv:.6f}")
                y_pos -= 25
            
            # Check for new page if needed
            if y_pos < 100:
                c.showPage()
                y_pos = height - 50
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y_pos, "Safety Reports (continued):")
                y_pos -= 30
                
        except Exception as e:
            c.setFont("Helvetica-Italic", 10)
            c.drawString(70, y_pos, f"Error processing {report_file.name}: {str(e)}")
            y_pos -= 20
    
    # Summary
    if y_pos < 150:
        c.showPage()
        y_pos = height - 50
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_pos, "Summary:")
    y_pos -= 20
    c.setFont("Helvetica", 10)
    c.drawString(70, y_pos, f"Total reports processed: {len(report_files)}")
    
    # Save the PDF
    c.save()
    print(f"PDF summary saved to {output_file}")
    return True

if __name__ == "__main__":
    import time
    success = generate_summary_pdf()
    sys.exit(0 if success else 1)
