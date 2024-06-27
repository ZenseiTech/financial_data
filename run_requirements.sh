
# run with source run_requirements.sh

pip freeze | xargs pip uninstall -y

pip install -r requirements.txt
