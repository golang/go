#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


bash clean.bash

cd 6l
bash mkenam
make enam.o
cd ..

echo; echo; echo %%%% making cc %%%%; echo
cd cc
make install
cd ..

echo; echo; echo %%%% making 6l %%%%; echo
cd 6l
make install
cd ..

echo; echo; echo %%%% making 6a %%%%; echo
cd 6a
make install
cd ..

echo; echo; echo %%%% making 6c %%%%; echo
cd 6c
make install
cd ..

echo; echo; echo %%%% making gc %%%%; echo
cd gc
make install
cd ..

echo; echo; echo %%%% making 6g %%%%; echo
cd 6g
make install
cd ..

echo; echo; echo %%%% making ar %%%%; echo
cd ar
make install
cd ..

echo; echo; echo %%%% making db %%%%; echo
cd db
make install
cd ..
