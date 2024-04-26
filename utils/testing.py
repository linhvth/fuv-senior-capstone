import inspect
from SGD import *
from datetime import datetime
import os
from globalVar import *
from misc import *

### Test function
def testing_opt(f, df, n_dims: int, true_global_min: int, init_point=None, 
                step_size=1e-2, max_iter=1000, n_streams=300, epsilon=1e-3, 
                tolerance=1e-20, save_result=True, name_result=None, 
                print_terminal=True):
  optimizer = SGD()
  optimizer.optimize(f, df, n_dims=n_dims, init_point=init_point, step_size=step_size, 
                      max_iter=max_iter, n_streams=n_streams, epsilon=epsilon, 
                      tolerance=tolerance)
  
  # Create result map
  result = dict()

  result['f'] = inspect.getsource(f).split("=", 1)[1].split(":", 1)[1][:-1]
  result['true solution'] = true_global_min
  result['init_point'] = optimizer.get_init_point()
  result['solution found by SGD'] = optimizer.get_solution()

  if save_result:
    # Create path to record result
    get_date = datetime.date.today().strftime('%Y%m%d')
    check_path_file(RESULT_PATH)
    filename = get_date + '_' + name_result if name_result is not None else '_untitled'
    save_path = os.path.join(RESULT_PATH, get_date + filename + '.txt')

    # Write to the result file
    write_as_txt(result, save_path)

  if print_terminal:
    print(result)
    print(f"f: {f(optimizer.get_solution())}")
    print(f"df: {df(optimizer.get_solution())}")