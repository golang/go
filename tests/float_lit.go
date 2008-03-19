// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
  [ 0.,
    +10.,
    -210.,
        
    .0,
    +.01,
    -.012,
       
    0.0,
    +10.01,
    -210.012,

    0E+1,
    +10e2,
    -210e3,
    
    0E-1,
    +0e23,
    -0e345,

    0E1,
    +10e23,
    -210e345,

    0.E1,
    +10.e+2,
    -210.e-3,
        
    .0E1,
    +.01e2,
    -.012e3,
       
    0.0E1,
    +10.01e2,
    -210.012e3,

    0.E+12,
    +10.e23,
    -210.e34,
        
    .0E-12,
    +.01e23,
    -.012e34,
       
    0.0E12,
    +10.01e23,
    -210.012e34,

    0.E123,
    +10.e+234,
    -210.e-345,
        
    .0E123,
    +.01e234,
    -.012e345,
       
    0.0E123,
    +10.01e234,
    -210.012e345
  ]
}
