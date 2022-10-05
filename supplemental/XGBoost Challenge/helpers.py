import pandas as pd

from IPython.display import HTML

def show_data_dictionary():
    
    """Returns embedded IDB Case Data Documentation PDF in notebook cell."""

    url = 'https://www.fjc.gov/sites/default/files/idb/codebooks/Civil%20Codebook%201988%20Forward.pdf'

    return HTML('<iframe src=%s width=700 height=350></iframe>' % url)

def show_problem_definition():
    
    """Returns embedded Legalist problem definition PDF in notebook cell."""

    url = './Problem%20Defintion.pdf'

    return HTML('<iframe src=%s width=700 height=350></iframe>' % url)

def show_assumptions():
    
    """Outputs my preliminary assumptions about the dataset."""
    
    columns = ['NOS',
           'JURY',
           'DEMANDED',
           'FILEJUDG',
           'FILEMAG',
           'TERMJUDG',
           'TERMMAG',
           'COUNTY',
           'TRANSOFF',
           'PROCPROG',
           'DISP',
           'AMTREC',
           'JUDGMENT',
           'TAPEYEAR']

    assumptions = ['code 422 is most relevant, as it corresponds to bankruptcy appellate cases',
                   'could be a useful feature',
                   'could have predictive power',
                   'could be useful as a feature',
                   'could be useful as a feature',
                   'could be useful as a feature',
                   'could be useful as a feature',
                   'could have predictive power',
                   'could have predictive power',
                   'this is important and i need to know how',
                   'could be a good alternative target',
                   'could be fun to try and predict',
                   'target for prediction',
                   'should probably filter out those with code 2099 ("pending")']
    
    assumptions_df = pd.DataFrame({})
    
    assumptions_df['column'] = pd.Series(columns)
    assumptions_df['assumption'] = pd.Series(assumptions)
    
    return assumptions_df