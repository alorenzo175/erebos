mkdir nc4
for FILE in $(find ./hdf4/*.hdf -type l);
do
    h4tonccf_nc4 $(ls -lh $FILE | awk '{print $11}') $(echo $FILE | sed -e 's/hdf/nc/g');
done;
