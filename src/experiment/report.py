from fpdf import FPDF


class Report:
    
    def write_report(self, fname):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 16)
        pdf.set_margins(0, 0)
        pdf.image('assets/header.png', x=0, y=0, w=pdf.epw)
        pdf.output(fname)