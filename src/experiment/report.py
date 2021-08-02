from fpdf import FPDF


class Report(FPDF):

    def __init__(self):
        super().__init__()
        self.set_font('helvetica', 'B', 16)
        self.set_margins(0, 0)
        self.add_page()

        self.add_font('kievitoffc', fname='assets/KievitOffc.ttf', uni=True)
        self.add_font('kievitoffc', style='B', fname='assets/KievitOffc-Bold.ttf', uni=True)
    
    def header(self):
        self.image('assets/header.png', x=0, y=0, w=self.epw)
        # self.cell(w=self.eph, h=20)

    def set_dataset_info(self, dataset_name: str):
        self.set_y(18)
        self.set_font('kievitoffc', size = 16)
        self.cell(w=(self.epw / 2), border=1, txt="Dataset Info", align='C', ln=2)

        self.set_font('kievitoffc', size = 12)
        self.cell(w=(self.epw / 2), border=1, txt=f"Name:   {dataset_name}", ln=2)
        self.cell(w=(self.epw / 2), border=1, txt="Number of Samples:\t", ln=2)


    def set_summary_stats(self):
        self.set_xy((self.epw / 2), 18)
        self.set_font('kievitoffc', size = 16)

        self.cell(w=(self.epw / 2), border=1, txt="Summary", align='C')

        