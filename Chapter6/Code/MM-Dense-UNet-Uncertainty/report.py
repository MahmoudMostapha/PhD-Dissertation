import scipy
import torch

def get_infos(TEST_EXP_ID, DATA_INFO):


    package_info = {
            'projectName':'MR Robustness',
            'projectDescription':''
            }

    test_exp_info = {
            'name': TEST_EXP_ID,
            'dataInfo': [
                    { 'name':'Dataset', 'type':'text', 'value':DATA_INFO['DATA']}                  
                ],
            'headers': [
                    { 'name':'ValidationContextLoss', 'type':'number'},
                    { 'name':'Input', 'type':'volume'},
                    { 'name':'Prediction', 'type':'volume'},
                    { 'name':'Target', 'type':'volume'},
                    { 'name':'SliceIns1', 'type':'image'},
                    { 'name':'SliceIns2', 'type':'image'},
                    { 'name':'SliceIns3', 'type':'image'},
                    { 'name':'SlicePred1', 'type':'image'},
                    { 'name':'SlicePred2', 'type':'image'},
                    { 'name':'SlicePred3', 'type':'image'},
                    { 'name':'SliceTgs1', 'type':'image'},
                    { 'name':'SliceTgs2', 'type':'image'},
                    { 'name':'SliceTgs3', 'type':'image'}
                ]
        }
    return package_info, test_exp_info

def data_preparer_tissue(self, ins, pred, tgs, vloss, metrics):
    res = {}

    ins = ins[0]

    pred = torch.argmax(pred, dim=1)
    tgs = torch.argmax(tgs, dim=1)

    res.update([self.report_value(k, v, 'number') for k, v in metrics.items()])
    res.update([self.report_value('Input', ins[0, 0], 'volume'),
               self.report_value('Target', tgs[0], 'volume'),
               self.report_value('Prediction', pred[0], 'volume')])

    idx1 = pred.size(1) // 4
    idx2 = 2 * pred.size(1) // 4
    idx3 = 3 * pred.size(1) // 4

    res.update([self.report_value('SliceIns1', ins[0, 0, idx1, :, :], 'image'),
               self.report_value('SliceIns2', ins[0, 0, idx2, :, :], 'image'),
               self.report_value('SliceIns3', ins[0, 0, idx3, :, :], 'image')])

    res.update([self.report_value('SlicePred1', pred[0, idx1, :, :], 'image'),
               self.report_value('SlicePred2', pred[0, idx2, :, :], 'image'),
               self.report_value('SlicePred3', pred[0, idx3, :, :], 'image')])

    res.update([self.report_value('SliceTgs1', tgs[0, idx1, :, :], 'image'),
               self.report_value('SliceTgs2', tgs[0, idx2, :, :], 'image'),
               self.report_value('SliceTgs3', tgs[0, idx3, :, :], 'image')])

    return res