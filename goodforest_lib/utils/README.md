# How to use Google Cloud Storage

## For MacOS

1. Install the correct CLI from (https://cloud.google.com/sdk/docs/install?hl=fr)[https://cloud.google.com/sdk/docs/install?hl=fr]
2. Go to terminal and execute `./google-cloud-sdk/install.sh`
3. Execute `./google-cloud-sdk/bin/gcloud init` to add `gcloud` to PATH
4. Add your credentials on your device: `gcloud auth application-default login`

You're good to go and use GCS!

## For Ubuntu (CentraleSupélec GPUs)

1. `sudo apt-get update`
2. `sudo apt-get install apt-transport-https ca-certificates gnupg curl`
3. `curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg`
5. `RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y`
6. `sudo gcloud init`
7. `gcloud auth application-default login`