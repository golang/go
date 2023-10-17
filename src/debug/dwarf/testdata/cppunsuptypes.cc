// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// cppunsuptypes.elf built with g++ 7.3
//    g++ -g -c -o cppunsuptypes.elf cppunsuptypes.cc

int i = 3;
double d = 3;

// anonymous reference type
int &culprit = i;

// named reference type
typedef double &dref;
dref dr = d;

// incorporated into another type
typedef struct {
  dref q;
  int &r;
} hasrefs;

hasrefs hr = { d, i };

// This code is intended to trigger a DWARF "pointer to member" type DIE
struct CS { int dm; };

int foo()
{
  int CS::* pdm = &CS::dm;
  CS cs = {42};
  return cs.*pdm;
}
