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
  e = 2.718281828;
)

type (
  Empty interface {};
  Point struct {
    x, y int;
  };
  Point2 Point
)
  
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
  for {}
  for {};
  for j := 0; j < i; j++ {
    if i == 0 {
    } else i = 0;
    var x float
  }
  foo: switch {
    case i < y:
    case i < j:
    case i == 0, i == 1, i == j:
      i++; i++;
    default:
      break
  }
}
