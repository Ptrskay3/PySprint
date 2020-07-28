PKG_NAME=pysprint
USER=ptrskay

mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld
conda build .


find $CONDA_BLD_PATH/ -name *.tar.bz2 | while read file
do
    echo $file
    anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER -l nightly $file --force
done
