✈️ Flight Phase Identification System (Offline + ML + LSTM)

A powerful offline system to automatically identify and label flight phases using machine learning. It processes FDR (Flight Data Recorder) data, clusters key flight parameters, and predicts flight phases with an LSTM model. Built with KMeans and TensorFlow, this system helps aviation analysts understand flight behavior without requiring an internet connection.

🔍 Features
🛬 Offline Flight Phase Detection using clustering (KMeans) and deep learning (LSTM).
📊 Supports CSV and tabular FDR data, extracting Time, Speed, and Altitude.
🧠 Learns and labels flight phases like Takeoff, Climb, Cruise, and Descent.
📈 3D Visualization of flight clusters for exploratory analysis.
🧮 Silhouette scoring to validate clustering quality.
🔁 LSTM model to learn phase transitions from time-series data.
🔒 Private & efficient — runs entirely on your local machine.
🎨 Training history plots to inspect model learning performance.
📂 Modular code — easy to extend for more parameters or different datasets.

