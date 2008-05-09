// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
  []int(
    0,
    123,
    0123,
    0000,
    0x0,
    0x123,
    0X0,
    0X123
  );
}
