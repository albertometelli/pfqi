from trlib.algorithms.algorithm_handler import Double_FQI_SSEP_Handler
import numpy as np
import json


def calculate_q_values(double_fqi_ss_ep_handler, file_name):
    assert isinstance(double_fqi_ss_ep_handler, Double_FQI_SSEP_Handler)
    q_max_list = []
    q_min_list = []
    persistences = double_fqi_ss_ep_handler.get_persistences()
    number_of_persistences = len(persistences)
    sa, q_functions = double_fqi_ss_ep_handler.get_data_and_qs()

    for i in range(number_of_persistences):
        vals1 = q_functions[i][0].values(sa)
        vals2 = q_functions[i][1].values(sa)
        vals = np.column_stack((vals1, vals2))
        maxq = np.amax(vals, axis=1)
        minq = np.amin(vals, axis=1)
        q_max_list.append(np.average(maxq).item())
        q_min_list.append(np.average(minq).item())

    output = [q_max_list, q_min_list]
    with open('Q_bounds' + file_name + '.json', 'w') as f:
        f.write(json.dumps(output))
