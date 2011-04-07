// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type (
       a interface{}
       b interface{}
)

func main() {
       x := a(1)
       z := b(x)
       _ = z
}
