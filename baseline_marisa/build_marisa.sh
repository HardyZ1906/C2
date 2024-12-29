#! /bin/bash

cd marisa
autoreconf -i
./configure --enable-static --enable-native-code
make
mv lib/marisa/.libs/libmarisa.a ..