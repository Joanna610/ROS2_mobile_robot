import pandas as pd
import os

locals = os.path.join('/home/joanna/ros2_gz_sim','test_data.csv')
# Wczytaj dwa pliki CSV
df1 = pd.read_csv(locals)

locals2 = os.path.join('/home/joanna/ros2_gz_sim/bag_folder20241116_155547','rosbag_data_20241116_155547.csv')
df2 = pd.read_csv(locals2)

# Połącz pliki w pionie (jeden pod drugim)
df_combined = pd.concat([df1, df2])

# Zapisz wynik do nowego pliku CSV
df_combined.to_csv(locals, index=False)
