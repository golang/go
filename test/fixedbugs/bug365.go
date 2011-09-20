// errchk $G $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// check that compiler doesn't stop reading struct def
// after first unknown type.

// Fixes issue 2110.

package main

type S struct {
	err os.Error  // ERROR "undefined|expected package"
	Num int
}

func main() {
	s := S{}
	_ = s.Num  // no error here please
}
