USER=ptrskay

set -e

array=( 3.6 3.7 3.8 )

mkdir ~/conda-bld
conda config --set anaconda_upload no
conda config --set ssl_verify no
export CONDA_BLD_PATH=~/conda-bld

for i in "${array[@]}"
do
	conda build . --python $i
done

find $CONDA_BLD_PATH/ -name *.tar.bz2 | while read file
do
    echo $file
    conda convert --platform all $file -o $CONDA_BLD_PATH/
done

find $CONDA_BLD_PATH/ -name *.tar.bz2 | while read file
do
    echo $file
    anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER $file --force
done
