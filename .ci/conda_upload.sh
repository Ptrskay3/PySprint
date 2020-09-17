USER=ptrskay

set -e

mkdir ~/conda-bld
conda config --set anaconda_upload no
conda config --set ssl_verify no
export CONDA_BLD_PATH=~/conda-bld
conda build . --python=3.8

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
