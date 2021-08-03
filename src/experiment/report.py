from preprocessing.datasetsummary import DatasetSummary
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

    def set_dataset_info(self, ds: DatasetSummary):
        self.set_y(18)
        self.set_font('kievitoffc', size = 16)
        self.cell(w=(self.epw / 2), border=1, txt="Dataset Info", align='C', ln=2)

        self.set_font('kievitoffc', size = 12)
        self.cell(w=(self.epw / 2), border=1, txt=f"Name:   {ds.get_songset_name()}", ln=2)
        self.cell(w=(self.epw / 2), border=1, txt=f"Number of Samples:    {ds.get_n_examples()}", ln=2)
        self.cell(w=(self.epw / 2), border=1, txt=f"Number of Features:    {ds.get_n_features()}", ln=2)
        self.cell(w=(self.epw / 2), border=1, txt=f"Number of Comments:    {ds.get_n_comments()}", ln=2)
        self.cell(w=(self.epw / 2), border=1, txt=f"Number of Words:    {ds.get_n_words()}", ln=2)


    def set_summary_stats(self):
        self.set_xy((self.epw / 2), 18)
        self.set_font('kievitoffc', size = 16)

        self.cell(w=(self.epw / 2), border=1, txt="Summary", align='C')

        