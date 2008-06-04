#!/bin/bash
# Copyright 2009 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


bash clean.bash

cd 6l
bash mkenam
mk enam.o
cd ..

echo; echo; echo %%%% making cc %%%%; echo
cd cc
mk install
cd ..

echo; echo; echo %%%% making 6l %%%%; echo
cd 6l
mk install
cd ..

echo; echo; echo %%%% making 6a %%%%; echo
cd 6a
mk install
cd ..

echo; echo; echo %%%% making 6c %%%%; echo
cd 6c
mk install
cd ..

echo; echo; echo %%%% making gc %%%%; echo
cd gc
mk install
cd ..

echo; echo; echo %%%% making 6g %%%%; echo
cd 6g
mk install
cd ..
