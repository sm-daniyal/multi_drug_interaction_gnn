# ==============================================================================
# Project: Multi-Drug Interaction Prediction using Graph Neural Networks
# Course: Project Phase - II (BIS786) | Batch 25 | JNNCE Shivamogga
# Guide: Mrs. Manasa S M, Assistant Professor, Dept. of IS&E
# Module: Clinical Pharmacotherapy & Drug Interaction Audit Report Generator
# Developer: sm-daniyal
# Copyright (c) 2026 sm-daniyal. All rights reserved.
# ==============================================================================

import os
from datetime import datetime

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


class ClinicalReportGenerator:
    """Generates hospital-grade clinical prescription audit reports with academic credits and watermarking."""

    def __init__(self):
        self.timestamp = datetime.now().strftime("%d-%b-%Y %I:%M %p IST")

    @staticmethod
    def _draw_page_decorations(canvas, doc):
        """Draws academic prototype watermark, header running line, and footer page numbers."""
        canvas.saveState()
        
        # 1. Subtle Academic Watermark (Centered, 45 degree rotation)
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 34)
        canvas.setFillColor(colors.HexColor("#f1f5f9"))
        canvas.translate(306, 396)
        canvas.rotate(45)
        canvas.drawCentredString(0, 25, "ACADEMIC RESEARCH PROTOTYPE")
        canvas.setFont("Helvetica-Bold", 13)
        canvas.setFillColor(colors.HexColor("#e2e8f0"))
        canvas.drawCentredString(0, -2, "NOT FOR INDEPENDENT CLINICAL PRESCRIPTION DECISIONS")
        canvas.restoreState()
        
        # 2. Running Header
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(colors.HexColor("#475569"))
        canvas.drawString(36, 756, "JNN COLLEGE OF ENGINEERING (JNNCE) | DEPARTMENT OF IS&E")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(576, 756, "PROJECT PHASE - II (BIS786) | BATCH 25")
        
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.setLineWidth(0.75)
        canvas.line(36, 750, 576, 750)
        
        # 3. Running Footer
        canvas.line(36, 38, 576, 38)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(36, 26, "CLINICAL DECISION SUPPORT SYSTEM -- RESEARCH EVALUATION REPORT -- NOT A MEDICAL REPLACEMENT")
        page_num = canvas.getPageNumber()
        canvas.drawRightString(576, 26, f"Page {page_num}")
        
        canvas.restoreState()

    def generate_pdf(self, prediction_results: dict, drug_metadata: dict, output_path: str, patient_info: dict = None, *args, **kwargs) -> str:
        """
        Compile and save an official clinical prescription interaction audit PDF report.
        """
        if not HAS_REPORTLAB:
            return "ReportLab library not available. Please install reportlab."

        p_info = patient_info or {
            "age": 65,
            "sex": "Male",
            "renal": "Normal (eGFR >= 60 mL/min)",
            "hepatic": "Normal Hepatic Function"
        }

        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=46,
            bottomMargin=46
        )

        styles = getSampleStyleSheet()
        
        # Custom Typography
        inst_title_style = ParagraphStyle(
            'InstTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=13,
            leading=16,
            textColor=colors.HexColor('#0f172a'),
            alignment=1,
            spaceAfter=2
        )
        
        doc_title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=11,
            leading=14,
            textColor=colors.HexColor('#1e40af'),
            alignment=1,
            spaceAfter=3
        )
        
        meta_sub_style = ParagraphStyle(
            'MetaSub',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=8,
            leading=11,
            textColor=colors.HexColor('#64748b'),
            alignment=1,
            spaceAfter=8
        )
        
        section_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading3'],
            fontName='Helvetica-Bold',
            fontSize=9.5,
            leading=12,
            textColor=colors.HexColor('#0f172a'),
            spaceBefore=8,
            spaceAfter=4
        )

        body_style = ParagraphStyle(
            'ReportBody',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=8,
            leading=11,
            textColor=colors.HexColor('#1e293b')
        )
        
        bold_body_style = ParagraphStyle(
            'BoldReportBody',
            parent=body_style,
            fontName='Helvetica-Bold'
        )

        story = []

        # 1. Institutional Document Header
        story.append(Paragraph("JNN COLLEGE OF ENGINEERING, SHIVAMOGGA", inst_title_style))
        story.append(Paragraph("CLINICAL PHARMACOTHERAPY & DRUG INTERACTION AUDIT REPORT", doc_title_style))
        story.append(Paragraph(
            f"Department of Information Science & Engineering | Project Phase - II (BIS786) | Report Generated: {self.timestamp}",
            meta_sub_style
        ))

        # 2. Academic Project Credits & Patient Metadata Box
        credits_html = (
            "<b>Project Title:</b> Multi-Drug Interaction Prediction using Graph Neural Networks<br/>"
            "<b>Faculty Guide:</b> Mrs. Manasa S M, Assistant Professor, Dept. of IS&E, JNNCE<br/>"
            "<b>Student Investigators:</b> Srujan Bhargav K (4JN23IS138), Swabhiman Y R (4JN23IS150), "
            "Syed M. Daniyal (4JN23IS155), Vishal S P (4JN23IS186) [Batch 25]"
        )
        patient_html = (
            f"<b>Patient Demographics:</b> Age: {p_info.get('age', 65)} Yrs | Sex: {p_info.get('sex', 'Male')}<br/>"
            f"<b>Renal Function:</b> {p_info.get('renal', 'Normal')}<br/>"
            f"<b>Hepatic Status:</b> {p_info.get('hepatic', 'Normal')}<br/>"
            "<b>Computational Engine:</b> GATv2 Attention Network (Stanford SNAP Decagon KG)"
        )
        
        meta_table_data = [
            [Paragraph("<b>ACADEMIC RESEARCH & INVESTIGATION CREDITS</b>", bold_body_style),
             Paragraph("<b>PATIENT PROFILE & SYSTEM METADATA</b>", bold_body_style)],
            [Paragraph(credits_html, body_style),
             Paragraph(patient_html, body_style)]
        ]
        meta_table = Table(meta_table_data, colWidths=[270, 270])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 8))

        # 3. Overall Regimen Verdict Banner
        overall_sev = prediction_results.get("severity", "Moderate").upper()
        risk_pct = prediction_results.get("risk_score", 0.60) * 100
        
        if overall_sev == "MAJOR":
            banner_bg = colors.HexColor('#fef2f2')
            banner_fg = colors.HexColor('#991b1b')
            action_text = "REGIMEN EVALUATION: MAJOR INTERACTION POTENTIAL -- IMMEDIATE CLINICAL REVIEW REQUIRED"
        elif overall_sev == "MODERATE":
            banner_bg = colors.HexColor('#fffbeb')
            banner_fg = colors.HexColor('#92400e')
            action_text = "REGIMEN EVALUATION: MODERATE INTERACTION POTENTIAL -- ROUTINE MONITORING RECOMMENDED"
        else:
            banner_bg = colors.HexColor('#f0fdf4')
            banner_fg = colors.HexColor('#166534')
            action_text = "REGIMEN EVALUATION: MINIMAL INTERACTION RISK DETECTED"

        verdict_p = Paragraph(
            f"<b>{action_text}</b><br/>"
            f"<font size=7.5>Composite GNN Risk Score: <b>{risk_pct:.1f}%</b> | "
            f"Evaluated across {len(prediction_results.get('pairwise', {}))} pairwise combination(s)</font>",
            ParagraphStyle('VerdictP', parent=body_style, textColor=banner_fg, alignment=1)
        )
        verdict_table = Table([[verdict_p]], colWidths=[540])
        verdict_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), banner_bg),
            ('BOX', (0, 0), (-1, -1), 1, banner_fg),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        story.append(verdict_table)
        story.append(Spacer(1, 6))

        # 4. Active Prescribed Regimen Profile Table
        story.append(Paragraph("1. PRESCRIBED ACTIVE MEDICATION REGIMEN", section_style))
        med_rows = [
            [
                Paragraph("<b>Medication (Generic)</b>", bold_body_style),
                Paragraph("<b>Pharmacological Class</b>", bold_body_style),
                Paragraph("<b>Primary Cellular Targets</b>", bold_body_style),
                Paragraph("<b>Metabolic Clearance (CYP450)</b>", bold_body_style)
            ]
        ]
        
        for sid, meta in drug_metadata.items():
            gen_name = meta.get("generic", sid)
            d_class = meta.get("drug_class", "Therapeutic Agent")
            targets = ", ".join(meta.get("targets", ["CYP3A4", "PTGS2"]))
            enzymes = ", ".join(meta.get("enzymes", ["CYP3A4"]))
            
            med_rows.append([
                Paragraph(f"<b>{gen_name}</b>", body_style),
                Paragraph(d_class, body_style),
                Paragraph(targets, body_style),
                Paragraph(enzymes, body_style)
            ])
            
        med_table = Table(med_rows, colWidths=[130, 130, 140, 140])
        med_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
            ('PADDING', (0, 0), (-1, -1), 4.5),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(med_table)
        story.append(Spacer(1, 6))

        # 5. Concomitant Interaction Findings & Pharmacological Mechanism
        story.append(Paragraph("2. CONCOMITANT INTERACTION FINDINGS & PHARMACOLOGICAL MECHANISMS", section_style))
        pairwise = prediction_results.get("pairwise", {})
        
        if not pairwise or overall_sev == "MINOR":
            zero_p = Paragraph("No statistically significant adverse drug-drug interactions detected between the evaluated agents.", body_style)
            zero_table = Table([[zero_p]], colWidths=[540])
            zero_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('PADDING', (0, 0), (-1, -1), 6)
            ]))
            story.append(zero_table)
        else:
            for pair_key, pdata in pairwise.items():
                d1 = pdata.get("drug1", "Drug 1").title()
                d2 = pdata.get("drug2", "Drug 2").title()
                psev = pdata.get("severity", "Moderate").upper()
                mechanism = pdata.get("mechanism", f"Competitive binding interaction between {d1} and {d2}.")
                shared = pdata.get("shared_targets", []) + pdata.get("shared_enzymes", [])
                shared_str = ", ".join(set(shared)) if shared else "General clearance pathway"
                rec = pdata.get("clinical_recommendation", "Monitor clinical parameters during concomitant therapy.")
                
                card_content = [
                    Paragraph(f"<b>COMBINATION: {d1} + {d2} &nbsp;|&nbsp; Severity Tier: {psev}</b>", bold_body_style),
                    Paragraph(f"<b>Organ & Clearance Mechanism:</b> {mechanism}", body_style),
                    Paragraph(f"<b>Shared Molecular Targets / Enzymes:</b> {shared_str}", body_style),
                    Paragraph(f"<b>Attending Clinician Guidance:</b> {rec}", body_style),
                ]
                
                pair_table = Table([[c] for c in card_content], colWidths=[540])
                pair_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
                    ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#94a3b8')),
                    ('PADDING', (0, 0), (-1, -1), 4.5),
                ]))
                story.append(pair_table)
                story.append(Spacer(1, 4))

        # 6. Quantitative Model Predictions (Side Effects)
        first_pair = next(iter(pairwise.values())) if pairwise else None
        if first_pair and first_pair.get("side_effects"):
            story.append(Spacer(1, 3))
            story.append(Paragraph("3. TOP PREDICTED ADVERSE EVENT PROBABILITIES (GAT ATTENTION HEADS)", section_style))
            se_table_data = [
                [
                    Paragraph("<b>Adverse Reaction Category</b>", bold_body_style),
                    Paragraph("<b>Model Predicted Probability</b>", bold_body_style),
                    Paragraph("<b>Investigational Risk Tier</b>", bold_body_style)
                ]
            ]
            for se in first_pair.get("side_effects", [])[:4]:
                prob = float(se.get('probability', 0.5))
                tier = "Elevated Risk" if prob > 0.6 else "Moderate Risk" if prob > 0.4 else "Low Risk"
                se_table_data.append([
                    Paragraph(se.get('name', 'Reaction').title(), body_style),
                    Paragraph(f"{prob * 100:.1f}%", body_style),
                    Paragraph(tier, body_style)
                ])
            se_table = Table(se_table_data, colWidths=[240, 150, 150])
            se_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('PADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(se_table)

        # 7. Safer Alternative Substitutions
        alternatives = prediction_results.get("alternatives", [])
        if alternatives:
            story.append(Spacer(1, 4))
            story.append(Paragraph("4. CLINICALLY SAFER THERAPEUTIC REGIMEN SUBSTITUTIONS", section_style))
            alt_data = [
                [
                    Paragraph("<b>Conflicting Agent</b>", bold_body_style),
                    Paragraph("<b>Suggested Substitute</b>", bold_body_style),
                    Paragraph("<b>Pharmacological Rationale</b>", bold_body_style)
                ]
            ]
            for alt in alternatives:
                alt_data.append([
                    Paragraph(alt.get("original_drug", ""), body_style),
                    Paragraph(f"<b>{alt.get('suggested_drug', '')}</b> ({alt.get('class', '')})", body_style),
                    Paragraph(alt.get("rationale", ""), body_style)
                ])
            alt_table = Table(alt_data, colWidths=[120, 150, 270])
            alt_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f766e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('PADDING', (0, 0), (-1, -1), 4.5),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            story.append(alt_table)

        # 8. Mandatory Academic & Regulatory Watermark Disclaimer Box
        story.append(Spacer(1, 8))
        disclaimer_text = (
            "<b>ACADEMIC PROTOTYPE NOTICE & CLINICAL DISCLAIMER:</b> This report is generated by a Graph Attention "
            "Network (GATv2) research prototype developed for academic evaluation in Project Phase - II (BIS786) "
            "at J.N.N. College of Engineering (JNNCE), Shivamogga. This tool is designed strictly as an investigational "
            "computational decision support aid. <b>IT DOES NOT CONSTITUTE INDEPENDENT MEDICAL ADVICE AND IS NOT A "
            "SUBSTITUTE FOR THE EVALUATION, DIAGNOSIS, OR PRESCRIPTION REVIEW OF A LICENSED PHYSICIAN OR REGISTERED "
            "PHARMACIST.</b> Clinical outcomes should always be verified against standard institutional pharmacotherapy "
            "formularies and verified medical references."
        )
        disc_p = Paragraph(disclaimer_text, ParagraphStyle('DiscStyle', parent=body_style, fontSize=7.2, leading=9.5, textColor=colors.HexColor('#78350f')))
        disc_table = Table([[disc_p]], colWidths=[540])
        disc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fef3c7')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#d97706')),
            ('PADDING', (0, 0), (-1, -1), 6)
        ]))
        story.append(disc_table)
        story.append(Spacer(1, 8))

        # 9. Verification & Signatures Block
        sign_table = Table([
            [Paragraph("<b>Attending Physician Review:</b> ________________________", body_style),
             Paragraph("<b>Clinical Pharmacist Review:</b> ________________________", body_style)],
            [Paragraph(f"<b>Date:</b> {self.timestamp.split()[0]} &nbsp;&nbsp;&nbsp;&nbsp; <b>Status:</b> [ ] Approved &nbsp; [ ] Modified", body_style),
             Paragraph("<b>Project Phase - II:</b> Batch 25, Dept. of IS&E, JNNCE", body_style)]
        ], colWidths=[270, 270])
        sign_table.setStyle(TableStyle([
            ('PADDING', (0, 0), (-1, -1), 3),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(sign_table)

        # Build Document with Watermark and Page Numbers
        doc.build(
            story,
            onFirstPage=self._draw_page_decorations,
            onLaterPages=self._draw_page_decorations
        )
        return output_path
