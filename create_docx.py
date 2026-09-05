from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import re

doc = Document()

# Set default font
style = doc.styles['Normal']
font = style.font
font.name = 'Calibri'
font.size = Pt(11)

# Helper functions
def add_heading_styled(doc, text, level):
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)
    return heading

def add_code_block(doc, code_text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(code_text)
    run.font.name = 'Consolas'
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)
    # Add shading
    shading = OxmlElement('w:shd')
    shading.set(qn('w:fill'), 'F4F4F4')
    shading.set(qn('w:val'), 'clear')
    p.paragraph_format.element.get_or_add_pPr().append(shading)
    return p

def add_table_from_markdown(doc, markdown_table):
    """Parse markdown table and add to document"""
    lines = markdown_table.strip().split('\n')
    if len(lines) < 3:
        return
    
    # Parse header
    headers = [cell.strip() for cell in lines[0].split('|')[1:-1]]
    # Parse separator (skip)
    # Parse rows
    rows = []
    for line in lines[2:]:
        if line.strip():
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if len(cells) == len(headers):
                rows.append(cells)
    
    if not rows:
        return
    
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # Header row
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = header
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(9)
    
    # Data rows
    for row_idx, row_data in enumerate(rows):
        for col_idx, cell_text in enumerate(row_data):
            cell = table.rows[row_idx + 1].cells[col_idx]
            cell.text = cell_text
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(9)

def parse_markdown_content(content):
    """Parse markdown content and add to document"""
    lines = content.split('\n')
    i = 0
    in_code_block = False
    code_buffer = []
    in_table = False
    table_buffer = []
    
    while i < len(lines):
        line = lines[i]
        
        # Handle code blocks
        if line.strip().startswith('```'):
            if not in_code_block:
                in_code_block = True
                code_buffer = []
            else:
                in_code_block = False
                add_code_block(doc, '\n'.join(code_buffer))
            i += 1
            continue
        
        if in_code_block:
            code_buffer.append(line)
            i += 1
            continue
        
        # Handle tables
        if '|' in line and line.strip().startswith('|') and line.strip().endswith('|'):
            if not in_table:
                in_table = True
                table_buffer = [line]
            else:
                table_buffer.append(line)
            i += 1
            continue
        else:
            if in_table:
                in_table = False
                add_table_from_markdown(doc, '\n'.join(table_buffer))
                table_buffer = []
        
        # Headings
        if line.startswith('# '):
            add_heading_styled(doc, line[2:].strip(), 0)
        elif line.startswith('## '):
            add_heading_styled(doc, line[3:].strip(), 1)
        elif line.startswith('### '):
            add_heading_styled(doc, line[4:].strip(), 2)
        elif line.startswith('#### '):
            add_heading_styled(doc, line[5:].strip(), 3)
        # Bold/italic text
        elif line.strip() == '---':
            doc.add_paragraph('').paragraph_format.space_after = Pt(12)
            # Add horizontal line
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(12)
            pPr = p.paragraph_format.element.get_or_add_pPr()
            pBdr = OxmlElement('w:pBdr')
            bottom = OxmlElement('w:bottom')
            bottom.set(qn('w:val'), 'single')
            bottom.set(qn('w:sz'), '6')
            bottom.set(qn('w:space'), '1')
            bottom.set(qn('w:color'), '2C3E50')
            pBdr.append(bottom)
            pPr.append(pBdr)
        # Image placeholders
        elif 'Image Placeholder' in line or '`.png`' in line or line.strip().startswith('- `'):
            p = doc.add_paragraph()
            run = p.add_run(line.strip())
            run.font.italic = True
            run.font.color.rgb = RGBColor(0x7F, 0x8C, 0x8D)
            run.font.size = Pt(10)
        # Regular paragraph
        elif line.strip():
            p = doc.add_paragraph()
            # Handle inline formatting
            parts = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`)', line)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    run = p.add_run(part[2:-2])
                    run.bold = True
                elif part.startswith('*') and part.endswith('*'):
                    run = p.add_run(part[1:-1])
                    run.italic = True
                elif part.startswith('`') and part.endswith('`'):
                    run = p.add_run(part[1:-1])
                    run.font.name = 'Consolas'
                    run.font.size = Pt(10)
                else:
                    run = p.add_run(part)
                run.font.size = Pt(11)
        else:
            # Empty line - add small space
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(4)
        
        i += 1
    
    # Handle any remaining table
    if in_table and table_buffer:
        add_table_from_markdown(doc, '\n'.join(table_buffer))

# Read markdown file
with open('report.md', 'r') as f:
    content = f.read()

# Parse and create document
parse_markdown_content(content)

# Save
doc.save('report.docx')
print("Report saved as report.docx")