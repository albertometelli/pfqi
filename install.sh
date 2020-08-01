function install_packages () {
	conda create -n $1 python=3.5 pip
	eval "$(conda shell.bash hook)"
	conda activate $1
    conda install pandas matplotlib scikit-learn
    conda install -c conda-forge swig
    pip install --upgrade pip
	pip install joblib
	pip install gym
    pip install joblib
    pip install box2d-py
	scr_dir=$(pwd)
	export scr_dir=$scr_dir
	conda develop $scr_dir
}

if [ "$#" -ne 1]; then
    echo "you have to specify the name of the environment"
else
	yes Y | install_packages $1
fi