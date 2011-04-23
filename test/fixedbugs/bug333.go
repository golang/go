// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 1709

package main

func main() {
       type Ts string
       var ts Ts
       _ = []byte(ts)
}

/*
bug333.go:14: cannot use ts (type Ts) as type string in function argument
*/
