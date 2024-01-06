conda install -y ipykernel
python -m ipykernel install --user --name $ENV_NAME --display-name "Python (CSCK_700)"

#install packages
pip install tensorflow[and-cuda] tensorrt tensorflow-addons
pip install -r requirements.txt

# #install graphviz
# #which distro are we on?
# distro=$(awk -F= '/^NAME/{print $2}' /etc/os-release)
# echo $distro
# #install graphviz
# if [ "$distro" == "\"Ubuntu\"" ]; then
#     echo "Installing graphviz on Ubuntu"
#     sudo apt-get install graphviz
# elif [ "$distro" == "\"CentOS Linux\"" ]; then
#     echo "Installing graphviz on CentOS"
#     sudo yum install graphviz
# elif [ "$distro" == "\"Fedora\"" ]; then
#     echo "Installing graphviz on Fedora"
#     sudo dnf install graphviz
# elif [ "$distro" == "\"Mac OS X\"" ]; then
#     echo "Installing graphviz on Mac"
#     brew install graphviz
# else
#     echo "Installing graphviz on Windows"
#     conda install graphviz
# fi

