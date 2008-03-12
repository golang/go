// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is test0.go.

package Test0

const
  a_const = 0
  
const (
  pi = /* the usual */ 3.14159265358979323;
  e = 2.718281828
)

type
  Point struct {
    x, y int
  }
  
var (
  x1 int;
  x2 int;
  u, v, w float
)

func foo() {}

func min(x, y int) int {
  if x < y { return x }
  return y
}

func swap(x, y int) (u, v int) {
  u = y;
  v = x;
  return
}

func control_structs() {
  i := 0;
  for {
    i++
  }
}
