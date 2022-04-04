
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import numpy as np
import os,sys

import GLTransformer as glt

from dataset import AgePredictionDataset


def load_samples(split_fnames, bin_width=1, min_bin_count=10, max_samples=None):
    import re
    import pandas as pd

    def correct_path(path):
        path = path.replace('/ABIDE/', '/ABIDE_I/')
        path = path.replace('/NIH-PD/', '/NIH_PD/')
        return path
    
    def get_ravens_path(path):
        path = path.replace('reg', 'ravens')
        return path
    
    def extract_dataset(path):
        name = re.match(r'/neuro/labs/grantlab/research/MRI_Predict_Age/([^/]*)', path)[1]
        return 'MGHBCH' if name in ('MGH', 'BCH') else name

    schema = {'id': str, 'age': float, 'sex': str, 'path': str}
    dfs = []
    for fname in split_fnames:
        split_num = int(fname[:-len('.list')].split('_')[-1])

        df = pd.read_csv(fname, sep=' ', header=None, names=['id', 'age', 'sex', 'path'], dtype=schema)
        df['path'] = df['path'].apply(correct_path)
        df['ravens_path'] = df['path'].apply(get_ravens_path)
        df['agebin'] = df['age'] // bin_width
        df['split'] = split_num
        df['dataset'] = df['path'].apply(extract_dataset)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, axis=0)
    
    if max_samples is not None:
        combined_df = combined_df.sample(max_samples)

    if min_bin_count is not None:
        bin_counts = combined_df['agebin'].value_counts()
        bins_below_cutoff = [bin for bin in bin_counts.keys() if bin_counts[bin] < min_bin_count]
        combined_df = combined_df[~combined_df['agebin'].isin(bins_below_cutoff)]
    
    # Filter out files that don't exist
    exists = combined_df['path'].apply(os.path.isfile)
    missing_files = combined_df['path'][~exists]
    print(f"{len(missing_files)} file(s) are missing:")
    print('\n'.join(missing_files))
    combined_df = combined_df[exists]
    
    return combined_df


class MRIage:
    def __init__(self):

        self.device = 'cuda'
            
        self.batch_size = 16
        
        print('using cuda:',self.device)
        print('batch size:',self.batch_size)
        
        self.learning_rate = 0.0001
        
        self.logdir = 'logsres/'
        if not os.path.exists(self.logdir):
            os.mkdir(self.logdir)
        
        self.subjectlogdir = 'subjectres/'
        if not os.path.exists(self.subjectlogdir):
            os.mkdir(self.subjectlogdir)
        
        folder='/neuro/labs/grantlab/research/MRI_Predict_Age/PythonCode/Extract2DSlice/TripleSet/data2dslice/'
        self.model_dir = 'save_models'
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
            
        basedir = 'e2-GLTransformer2D-slice80'
        
        self.trn_logfile = self.logdir + basedir +'_train.log'
        self.txt_txtfile = self.logdir + basedir +'_test.log'
        self.txt_txtfile_subjects = self.subjectlogdir + basedir +'_test_subjects'
        self.eval_txtfile_subjects = self.subjectlogdir + basedir +'_eval'
        
        self.modelfile = basedir

        # Load the dataset

        import glob
        from sklearn.model_selection import train_test_split

        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        SPLITS_DIR = os.path.join(ROOT_DIR, "folderlist")

        split_fnames = glob.glob(f"{SPLITS_DIR}/nfold_imglist_all_nfold_*.list")
        assert len(split_fnames) == 5

        df = load_samples(split_fnames)
        
        _train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['agebin'])
        #train_df = resample(_train_df, mode=opts.sample, undersamp_limit=opts.undersamp_limit, oversamp_limit=opts.oversamp_limit)
        train_df = _train_df

        train_set = AgePredictionDataset(train_df)
        
        self.training_data_loader = DataLoader(dataset=train_set,num_workers=0,
                                               batch_size = self.batch_size,shuffle=True)

        test_set = AgePredictionDataset(val_df)
        
        self.testing_data_loader = DataLoader(dataset=test_set,num_workers=0,
                                               batch_size = self.batch_size,shuffle=False)
        
        indim1 = 130
        
        self.model = glt.GlobalLocalBrainAge(inplace=indim1,
                            num_classes=1).to(self.device)
      
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.learning_rate)
        self.scheduler = lr_scheduler.StepLR(self.optimizer,step_size=25,gamma=0.5)
        
    def write_log(self,msg,logfile):
        with open(logfile,'a') as fp:
            fp.write('%s\n'%msg)
            
    
    def train(self,epoch):
        self.model.train()

        losstotal = []
        
        for iters,batch in enumerate(self.training_data_loader,1):
            
            image = batch[0].to(self.device).float()
            age = batch[1].to(self.device).float()
            
            self.optimizer.zero_grad()
            
            logitslist = self.model(image)
            
            loss = 0
            for idx,logits in enumerate(logitslist):
                loss += torch.mean(torch.abs(logits.squeeze(1)-age))
            loss.backward()
            
            self.optimizer.step()
            
            losstotal.append(loss.item())
            
        totalmean = np.mean(losstotal)
        msg = 'Epoch: %d (iters:%d) has train loss: %f '%(epoch,len(losstotal),totalmean)
      
        #print(msg)
        self.write_log(msg,self.trn_logfile)
        
    def test(self,epoch,testing_loader,logfile):
        self.model.eval()

        errlist = {}
        
        for iters,batch in enumerate(testing_loader,1):
            
            image = batch[0].to(self.device).float()
            age = batch[1].to(self.device).float()
            
            with torch.no_grad():
                logitslist = self.model(image)
                
            for idx,logits in enumerate(logitslist):
                predout=logits.squeeze(1)
                error1 = torch.abs(predout-age)
                error1 = error1.detach().cpu().numpy()
                error1 = list(error1)
                
                if idx not in errlist:
                    errlist[idx]=[]
                    
                errlist[idx] += error1
            
        msg = 'Epoch: %d (number:%d) has average error:'%(epoch,len(errlist[0]))
        for idx in range(len(errlist)):
            msg += ' %d:%.3f '%(idx,np.mean(np.abs(errlist[idx])))
        self.write_log(msg,logfile)

    def checkpoint(self,epoch):
        model_out_path = self.model_dir + '/' + self.modelfile + '-model-epoch-{}.path'.format(epoch)
        torch.save(self.model.state_dict(),model_out_path)
		
    def load_model(self,epoch):
        model_out_path = self.model_dir + '/' + self.modelfile + '-model-epoch-{}.path'.format(epoch)
        self.model.load_state_dict(torch.load(model_out_path,map_location=self.device))

    def train_loops(self,start,nepoch):
        if start > 0: 
            self.load_model(start-1)
        
        for epoch in range(start):
            self.scheduler.step()
            
        for epoch in range(start+1,nepoch+1):
            self.train(epoch)
            self.scheduler.step()
            
    
            self.test(epoch,self.testing_data_loader,self.txt_txtfile)
            
        self.checkpoint(nepoch)
            

if __name__ == '__main__':
    

    modeltrn = MRIage()
    modeltrn.train_loops(0, 80)
            
