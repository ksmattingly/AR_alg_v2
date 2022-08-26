import os
import hjson

# Import utils
# - raw IVT calc
# - IVT PR calc
# - 1000-700 hPa mean wind calc
# - regridding?


def run_AR_ID():
    _code_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = _code_dir+'/AR_ID_config.hjson'
    with open(config_path) as f:
        config = hjson.loads(f.read())
        
    # Continue with workflow, using parameters in the config dictionary
                
if __name__ == '__main__':
    run_AR_ID()