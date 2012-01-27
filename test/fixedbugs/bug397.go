// errchk $G -e $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Issue 2623
var m = map[string]int {
	"abc":1,
	1:2, // ERROR "cannot use 1.*as type string in map key|incompatible type"
}
