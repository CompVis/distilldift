mkdir -p data
cd data/
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar  -xf CUB_200_2011.tgz -C .
rm CUB_200_2011.tgz