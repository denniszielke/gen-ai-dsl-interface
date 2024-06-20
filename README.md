# Using Generative AI to create DSL
This project demonstrates how to teach a LLM the usage of a domain specific language to solve problems and use tools.

## Quickstart

The recommended way to execute the deployment is to fork the repo and then start a GitHub Codespace because there you will have all the tools to deploy the resources, test the code and try out the whole story end-to-end.
The project resources can be deployed into the following Azure regions:
- eastus

```

echo "log into azure dev cli - only once"
azd auth login

echo "provisioning all the resources with the azure dev cli"
azd up

echo "get and set the value for AZURE_ENV_NAME"
source <(azd env get-values | grep AZURE_ENV_NAME)

echo "building and deploying the streamlit user interface"
bash ./azd-hooks/deploy.sh web $AZURE_ENV_NAME

```

## Run the app locally

Navigate to the folder */src/web*.

```

cd /src/web

echo "installing python packages"
pip install -r requirements.txt

echo "starting app"
python -m streamlit run app.py --server.port=8000

```