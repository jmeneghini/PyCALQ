import logging
import os, glob
import yaml
import tqdm
from multiprocessing import Process

import sigmond
import fvspectrum.sigmond_util as sigmond_util
import general.plotting_handler as ph

doc = '''
preview_corrs - a task a read in and estimate/plot any Lattice QCD temporal correlator data files given

inputs
-------------
general:
  ensemble_id: cls21_c103       #required
  project_dir: /latticeQCD/raid3/sarahski/lqcd/C103_R005/test_pycalq_project #required
  sampling_info:                #not required 
    mode: Jackknife               #default Jackknife
  tweak_ensemble:               #not required
    omissions: []               #default []
    rebin: 1                    #default 1
preview_corrs:                  #required
  raw_data_files:               #required 
  - /latticeQCD/raid3/ahanlon/data/cls21_c103/updated_stats/sigmond.fwd/cls21_c103/nucleon_S0.bin
  create_pdfs: true             #not required #default true
  create_pickles: true          #not required #default true
  create_summary: true          #not required #default true
  figheight: 6                  #not required #default 6
  figwidth: 8                   #not required #default 8
  info: true                    #not required #default false
  plot: true                    #not required #default true
  generate_estimates: true              #not required #default true
'''
      
class SigmondPreviewCorrs:

    @property
    def info(self):
        return doc
    
    #initialize
    def __init__(self, task_name, proj_file_handler, general_params, task_params, sph):
        self.task_name = task_name
        self.project_handler = sph
        self.proj_file_handler = proj_file_handler

        if not task_params:
            logging.critical(f"No directory to view. Add 'raw_data_files' to '{task_name}' task parameters.")

        #check that raw_data_files are real files and not in project
        raw_data_files = []
        if 'raw_data_files' in task_params.keys():
            raw_data_files = task_params['raw_data_files']
        raw_data_files = sigmond_util.check_raw_data_files(raw_data_files, general_params['project_dir'])

        # self.project_info = sigmond_util.setup_project(general_params,raw_data_files)
        
        #other params
        self.other_params = {
            'generate_estimates': True,
            'create_pdfs': True,
            'create_pickles': True,
            'create_summary': True,
            'plot': True,
            'figwidth':8,
            'figheight':6,
            'separate_mom': True
        }
        sigmond_util.update_params(self.other_params,task_params) #update other_params with task_params, 
                                                                        #otherwise fill in missing task params

        if not self.other_params['create_pdfs'] and not self.other_params['create_pickles'] and not self.other_params['create_summary']:
            self.other_params['plot'] = False
        
        #make yaml output
        logging.info(f"Full input written to '{os.path.join(proj_file_handler.log_dir(), 'full_input.yml')}'.")
        with open( os.path.join(proj_file_handler.log_dir(), 'full_input.yml'), 'w+') as log_file:
            yaml.dump({"general":general_params, task_name: task_params}, log_file)

        self.project_handler.add_raw_data(raw_data_files)

        self.data_handler = self.project_handler.data_handler
        self.channels = self.data_handler.raw_channels[:]
        final_channels = sigmond_util.filter_channels(task_params, self.channels)
        remove_channels = set(self.channels)-set(final_channels)
        self.project_handler.remove_raw_data_channels(remove_channels)
        self.channels = final_channels

    def run(self):
        mcobs_handler, mcobs_get_handler = sigmond_util.get_mcobs_handlers(self.data_handler,self.project_handler.project_info)

        ##get operators -> print to logfile
        log_path = os.path.join(self.proj_file_handler.log_dir(), 'ops_log.yml')
        ops_list = {}
        ops_list["channels"] = {str(channel):{"operators":[str(op) for op in self.data_handler.getChannelOperators(channel)]} for channel in self.data_handler.raw_channels }
        
        logging.info(f"Channels and operators list written to '{log_path}'.")
        with open(log_path, 'w+') as log_file:
            yaml.dump(ops_list, log_file)
        
        save_to_self = not self.other_params['generate_estimates'] and self.other_params['plot']
        if save_to_self:
            self.data = {}
        
        if not self.other_params['generate_estimates'] and not self.other_params['plot']:
            logging.warning("You have set 'generate_estimates' to 'False' and 'plot' to 'False' thus making this task obsolete. Congrats.")
            return

        self.moms = []
        logging.info(f"Saving correlator estimates to directory {self.proj_file_handler.data_dir()}...")
        for channel in self.channels:
            self.moms.append(channel.psq)
            if save_to_self:
                self.data[channel] = {}
            for op1 in self.data_handler.getChannelOperators(channel):
                if save_to_self:
                    self.data[channel][op1] = {}
                for op2 in self.data_handler.getChannelOperators(channel):
                    corr = sigmond.CorrelatorInfo(op1.operator_info,op2.operator_info)
                    corr_name = repr(corr).replace(" ","-")
                    estimates = sigmond.getCorrelatorEstimates(mcobs_handler,corr,self.project_handler.hermitian,self.project_handler.subtract_vev,
                                                               sigmond.ComplexArg.RealPart, self.project_handler.project_info.sampling_info.getSamplingMode())
                    if save_to_self:
                        self.data[channel][op1][op2] = {}
                        self.data[channel][op1][op2]["corr"] = sigmond_util.estimates_to_df(estimates)
                    else:
                        sigmond_util.estimates_to_csv(estimates, self.proj_file_handler.corr_estimates_file(corr_name) )
                    estimates = sigmond.getEffectiveEnergy(mcobs_handler,corr,self.project_handler.hermitian,self.project_handler.subtract_vev,
                                                           sigmond.ComplexArg.RealPart, self.project_handler.project_info.sampling_info.getSamplingMode(),
                                                           self.project_handler.time_separation,self.project_handler.effective_energy_type,
                                                           self.project_handler.vev_const)
                    if save_to_self:
                        self.data[channel][op1][op2]["effen"] = sigmond_util.estimates_to_df(estimates)
                    else:
                        sigmond_util.estimates_to_csv(estimates, self.proj_file_handler.effen_estimates_file(corr_name) )

    def plot(self):
        #make plot for each correlator -> save to pickle and pdf
        if self.other_params['plot']:
            logging.info(f"Saving plots to directory {self.proj_file_handler.plot_dir()}...")
        else:
            logging.info(f"No plots requested.")
            return
        
        plh = ph.PlottingHandler()

        #set up fig object to reuse
        plh.create_fig(self.other_params['figwidth'], self.other_params['figheight'])
        
        #loop through same channels #make loading bar
        if self.project_handler.nodes:
            chunk_size = int(len(self.channels)/self.project_handler.nodes)+1
            channels_per_node = [self.channels[i:i + chunk_size] for i in range(0, len(self.channels), chunk_size)]
        if not self.other_params['generate_estimates'] and self.other_params['plot']:
            if self.project_handler.nodes:
                processes = []
                for channels in channels_per_node:
                    processes.append(Process(target=self.write_channel_plots_data,args=(channels, plh,)))
                    processes[-1].start()
                for process in processes:
                    process.join()
            else:
                for channel in tqdm.tqdm(self.channels):
                    self.write_channel_plots_data(channel, plh)
        else:
            if self.project_handler.nodes:  
                processes = []
                for channels in channels_per_node:
                    processes.append(Process(target=self.write_channel_plots,args=(channels, plh,)))
                    processes[-1].start()
                for process in processes:
                    process.join()
            else:
                for channel in tqdm.tqdm(self.channels):
                    self.write_channel_plots(channel, plh)

        if self.other_params['create_summary']:
            plh.create_summary_doc("Preview Data")
            if self.other_params['separate_mom']:
                self.moms = list(set(self.moms))
                for i in self.moms[1:]:
                    plh.create_summary_doc("Preview Data")
            for channel in self.channels:
                index = 0
                if self.other_params['separate_mom']:
                    index = self.moms.index(channel.psq)
                plh.append_section(str(channel), index)
                for op1 in self.data_handler.getChannelOperators(channel):
                    for op2 in self.data_handler.getChannelOperators(channel):
                        corr = sigmond.CorrelatorInfo(op1.operator_info,op2.operator_info)
                        corr_name = repr(corr).replace(" ","-")
                        #check that files exist
                        plh.add_correlator_subsection(repr(corr),self.proj_file_handler.corr_plot_file( corr_name, "pdf"),
                                                        self.proj_file_handler.effen_plot_file( corr_name, "pdf"), index)

            for f in glob.glob(self.proj_file_handler.summary_file('*')+".*"):
                os.remove(f)
            for f in glob.glob(self.proj_file_handler.summary_file()+".*"):
                os.remove(f)
            if self.other_params['separate_mom']:
                if self.project_handler.nodes:
                    processes = []
                    ip = 0
                    for i,psq in enumerate(self.moms):
                        if len(processes)==i:
                            processes.append(Process(target=plh.compile_pdf,args=(self.proj_file_handler.summary_file(psq),i,)))
                            processes[-1].start()
                        else:
                            processes[ip].join()
                            processes[ip] = Process(target=plh.compile_pdf,args=(self.proj_file_handler.summary_file(psq),i,))
                            processes[ip].start()
                        ip += 1
                        ip = ip%self.project_handler.nodes
                    for process in processes:
                        process.join()

                else:
                    loglevel = logging.getLogger().getEffectiveLevel()
                    logging.getLogger().setLevel(logging.WARNING)
                    for i,psq in enumerate(self.moms):
                        plh.compile_pdf(self.proj_file_handler.summary_file(psq),i) 
                    logging.getLogger().setLevel(loglevel)
                logging.info(f"Summary files saved to {self.proj_file_handler.summary_file('*')}.pdf.")
            else:
                plh.compile_pdf(self.proj_file_handler.summary_file()) 
                logging.info(f"Summary file saved to {self.proj_file_handler.summary_file()}.pdf.")

    def write_channel_plots_data(self, channels, plh):
        if type(channels)==list:
            for channel in channels:
                sigmond_util.write_channel_plots(self.data_handler.getChannelOperators(channel), plh, self.other_params['create_pickles'],
                            self.other_params['create_pdfs'] or self.other_params['create_summary'],self.proj_file_handler,
                            self.data[channel])
        else:
            channel = channels
            sigmond_util.write_channel_plots(self.data_handler.getChannelOperators(channel), plh, self.other_params['create_pickles'],
                            self.other_params['create_pdfs'] or self.other_params['create_summary'],self.proj_file_handler,
                            self.data[channel])
            
    def write_channel_plots(self, channels, plh):
        if type(channels)==list:
            for channel in channels:
                sigmond_util.write_channel_plots(self.data_handler.getChannelOperators(channel), plh, self.other_params['create_pickles'],
                            self.other_params['create_pdfs'] or self.other_params['create_summary'],self.proj_file_handler)
        else:
            channel = channels
            sigmond_util.write_channel_plots(self.data_handler.getChannelOperators(channel), plh, self.other_params['create_pickles'],
                            self.other_params['create_pdfs'] or self.other_params['create_summary'],self.proj_file_handler)

