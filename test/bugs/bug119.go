// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG: should not fail

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
  a := new(chan bool);
  for {
    select {
    case <- a:
      panic();
    default:
      break;
    }
    panic();
  }
}
