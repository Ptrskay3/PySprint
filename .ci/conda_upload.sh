PKG_NAME=pysprint
USER=ptrskay

set -e

mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld

ls -l
conda build .

find $CONDA_BLD_PATH/ -name *.tar.bz2 | while read file
do
    echo $file
    anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER $file --force
done
