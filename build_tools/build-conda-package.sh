#!/bin/bash

pkg='pysprint'
array=( 3.6 3.7 )

echo "Building conda package ..."
cd ~
conda skeleton pypi --version 0.10.0 $pkg
cd $pkg
cd ~

for i in "${array[@]}"
do
	conda-build --python $i $pkg
done

cd ~
platforms=( osx-64 linux-32 linux-64 win-32 win-64 )
find $HOME/conda-bld/linux-64/ -name *.tar.bz2 | while read file
do
    echo $file
    # conda convert --platform all $file  -o $HOME/conda-bld/
    for platform in "${platforms[@]}"
    do
       conda convert --platform $platform $file  -o $HOME/conda-bld/
    done
    
done

find $HOME/conda-bld/ -name *.tar.bz2 | while read file
do
    echo $file
    anaconda upload $file
done

echo "Done"