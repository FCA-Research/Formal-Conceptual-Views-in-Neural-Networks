mkdir image-data
mkdir image-data/fruit360
git clone https://github.com/Horea94/Fruit-Images-Dataset.git 
cd Fruit-Images-Dataset
git checkout 9ce036635e3d9608985231c6579870ecc482a5b2
cd ..
mv Fruit-Images-Dataset/Test image-data/fruit360
mv Fruit-Images-Dataset/Training image-data/fruit360
rm -rf Fruit-Images-Dataset
