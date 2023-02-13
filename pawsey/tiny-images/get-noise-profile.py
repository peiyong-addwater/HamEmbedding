from qiskit_ibm_provider import IBMProvider
from qiskit_aer.noise import NoiseModel
import time
import pickle
from datetime import timedelta

ts_curr = 0
data_count = 0
total_data = 100
query_interval = 6*3600

while total_data>=data_count:
    PROVIDER = IBMProvider()
    backend = PROVIDER.get_backend('ibm_perth')
    properties = backend.properties()
    noise_model = NoiseModel.from_backend(backend)
    #print("New calibration at: ", properties.to_dict()['last_update_date'])
    prop_date = properties.to_dict()['last_update_date']
    ts_prop = prop_date.timestamp()
    if ts_prop >= ts_curr:
        print("New calibration at: ", properties.to_dict()['last_update_date'])
        print("Saving noise model...")
        #noise_dict = noise_model.to_dict()
        #prop_dict = properties.to_dict()
        filename_noisemodel = f'ibm_perth-{prop_date.strftime("%Y%m%d-%H%M%S")}.noise_model'
        filename_dev_properties = f'ibm_perth-{prop_date.strftime("%Y%m%d-%H%M%S")}.dev_prop'
        with open(filename_noisemodel, 'wb') as f:
            pickle.dump(noise_model, f)
            print("Noise Model saved to:\n", filename_noisemodel)
        with open(filename_dev_properties, 'wb') as f:
            pickle.dump(properties, f)
            print("Device Properties saved to:\n", filename_dev_properties)
        data_count = data_count + 1
        ts_curr = ts_prop

    print(f"Waiting for {timedelta(seconds=query_interval)} until next query...")
    time.sleep(query_interval)