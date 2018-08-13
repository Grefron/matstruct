import win32com.client
# import pythoncom

#http://joshuaredstone.blogspot.com/2015/02/solidworks-macros-via-python.html


class SolidWorks:
    def __init__(self, year=None):
        if year is None:
            self.sw = win32com.client.Dispatch("SldWorks.Application")
        else:
            version = year - 1992
            self.sw = win32com.client.Dispatch("SldWorks.Application.{}".format(version))

    @property
    def active_document(self):
        return self.sw.ActiveDoc

    @property
    def selection_manager(self):
        return self.active_document.SelectionManager

    @property
    def feature_manager(self):
        return self.active_document.FeatureManager

    @property
    def sketch_manager(self):
        return self.active_document.SketchManager

    @property
    def equation_manager(self):
        return self.active_document.GetEquationMgr

    def select_by_id2(self, name, type_, x, y, z, do_append=True, mark=0):

sw = SolidWorks(2017)

print(sw.sw)


