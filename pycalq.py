import logging

import general.task_manager as tm
import general.config_handler as ch
import general.project_directory as pd

import fvspectrum.sigmond_project_handler as sph

# Import the new task classes directly
from fvspectrum.tasks.preview_correlators import CorrelatorPreviewTask
from fvspectrum.tasks.average_correlators import AverageCorrelatorsTask
from fvspectrum.tasks.rotate_correlators import RotateCorrelatorsTask
from fvspectrum.tasks.fit_spectrum import FitSpectrumTask
from fvspectrum.tasks.compare_spectrums import CompareSpectrumsTask

# Thanks to Drew and https://stackoverflow.com/a/48201163/191474
#ends code when run logging.error(message) or logging.critical(message)
# logging.warning(message) and logging.debug(message) won't end code but will output at certain verbosity settings
class ExitOnExceptionHandler(logging.StreamHandler):

  def emit(self, record):
    super().emit(record)
    if record.levelno in (logging.ERROR, logging.CRITICAL):
      raise RuntimeError
    
#set up logger
logging.basicConfig(format='%(levelname)s: %(message)s', handlers=[ExitOnExceptionHandler()], level=logging.INFO)

# set up the tasks config and order and classes
DEFAULT_TASKS = { #manage default configurations
    "tasks":
        {
            tm.Task.preview: None,
            tm.Task.average: None,
            tm.Task.rotate: None,
            tm.Task.fit: None,
            tm.Task.compare: None,
        }
}
                    
#manage order of tasks  
TASK_ORDER = list(dict(tm.Task.__members__).values())   
TASK_NAMES = {task.name:task for task in TASK_ORDER}
SIGMOND_TASKS = [ #manage which classes to use for each unique task -> change for selection (fvspectrum)
    tm.Task.preview,
    tm.Task.average,
    tm.Task.rotate,
    tm.Task.fit,
]                            
TASK_MAP = { #manage which classes to use for each unique task -> change for selection (fvspectrum)
    tm.Task.preview: CorrelatorPreviewTask,
    tm.Task.average: AverageCorrelatorsTask,
    tm.Task.rotate: RotateCorrelatorsTask,
    tm.Task.fit: FitSpectrumTask,
    tm.Task.compare: CompareSpectrumsTask,
}
TASK_DOC = { #imports documentation from each task
    tm.Task.preview: getattr(CorrelatorPreviewTask, 'info', 'Preview correlator data'),
    tm.Task.average: getattr(AverageCorrelatorsTask, 'info', 'Average correlators'),
    tm.Task.rotate: getattr(RotateCorrelatorsTask, 'info', 'Rotate correlators using GEVP'),
    tm.Task.fit: getattr(FitSpectrumTask, 'info', 'Fit spectrum to extract energies'),
    tm.Task.compare: getattr(CompareSpectrumsTask, 'info', 'Compare different spectrum analyses'),
}

#set required general parameters 
#items in list must be str or {str: list of str}
#we don't really want to go deeper than two levels
REQUIRED_GENERAL_CONFIGS = [
   'project_dir',
   'ensemble_id',
]


class PyCALQ:

    def __init__( self, general_configs, task_configs = DEFAULT_TASKS ):
        #set up config handler
        self.general_configs_handler = ch.ConfigHandler(general_configs)
        self.task_configs_handler = ch.ConfigHandler(task_configs)

        #perform checks on configs
        self.general_configs_handler.check('general',REQUIRED_GENERAL_CONFIGS)
        self.task_configs_handler.check('tasks',[])
        
        self.general_configs = self.general_configs_handler.configs['general']
        self.task_configs = self.task_configs_handler.configs['tasks']

        if type(self.task_configs)!=list:
            logging.error("Error in task configuration file. List tasks.")

        self.proj_dir = pd.ProjectDirectoryHandler(self.general_configs['project_dir'],TASK_ORDER)

        #sort tasks by task order
        task_names = [task.name for task in TASK_ORDER]
        self.task_configs.sort(key = lambda x: task_names.index(list(x.keys())[0]) if list(x.keys())[0] in task_names else len(self.task_configs))

        #set up sigmond handler and set up project directory handler 
        max_task = 0
        self.sig_proj_hand = None
        sigmond_tasks = []
        for task in self.task_configs:
            key = list(task.keys())
            if len(key)>1:
                logging.error("Error in task configuration file.")
            if key[0] in TASK_NAMES.keys():
                if TASK_NAMES[key[0]] in SIGMOND_TASKS:
                    sigmond_tasks.append(TASK_NAMES[key[0]])
                max_task = task_names.index(key[0]) if task_names.index(key[0])>max_task else max_task
        if sigmond_tasks:
            self.sig_proj_hand = sph.SigmondProjectHandler(self.general_configs,sigmond_tasks)

        for task in TASK_ORDER[:max_task+1]:
            self.proj_dir.set_task(task.value, task.name)

    def run( self ):
        #perform the tasks in TASK_ORDER
        for task_config in self.task_configs:
            task_name = list(task_config.keys())[0] #root
            if task_name in TASK_NAMES:
                task = TASK_NAMES[task_name]
                self.proj_dir.set_task(task.value, task.name)

                #print documentation if requested
                if 'info' in task_config[task.name]:
                   if task_config[task.name]['info']:
                      print(TASK_DOC[task])

                #if sigmond task, set up project handler
                logging.info(f"Setting up task: {task.name}...")
                if task in SIGMOND_TASKS:
                    this_task = TASK_MAP[task](task.name, self.proj_dir, self.general_configs,task_config[task.name], self.sig_proj_hand) #initialize
                else:
                    this_task = TASK_MAP[task](task.name, self.proj_dir, self.general_configs,task_config[task.name]) #initialize
                logging.info(f"Task {task.name} set up.")

                #perform the task, produce the data
                logging.info(f"Running task: {task.name}...")
                this_task.run() 
                logging.info(f"Task {task.name} completed.")

                #do not plot if "plot: False" is present in task yaml
                if 'plot' in task_config[task.name]:
                    if not task_config[task.name]['plot']:
                        if task in SIGMOND_TASKS:
                            self.sig_proj_hand.switch_tasks()
                        continue
                   
                logging.info(f"Plotting task: {task.name}...")
                this_task.plot() 
                logging.info(f"Task {task.name} plotted.")
                
                #handle memory of sigmond tasks
                if task in SIGMOND_TASKS:
                   self.sig_proj_hand.switch_tasks()





