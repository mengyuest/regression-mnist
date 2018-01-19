mkdir -p result/err
mkdir -p result/plots
mkdir -p data
cd data

if ! [ -f train-images-idx3-ubyte ]
then
    wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz 
    gunzip train-images-idx3-ubyte.gz
fi

if ! [ -f train-labels-idx1-ubyte ]
then
    wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz 
    gunzip train-labels-idx1-ubyte.gz
fi

if ! [ -f t10k-images-idx3-ubyte ]
then
    wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz 
    gunzip t10k-images-idx3-ubyte.gz
fi

if ! [ -f t10k-labels-idx1-ubyte ]
then
    wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    gunzip t10k-labels-idx1-ubyte.gz
fi

cd ..

echo "---------------------------------------"
echo "Finished"
echo "use 'python main.py' to execute program"


