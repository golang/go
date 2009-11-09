// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main
type I int
type S struct { f map[I]int }
var v1 = S{ make(map[int]int) }		// ERROR "cannot|illegal|incompatible|wrong"
var v2 map[I]int = map[int]int{}	// ERROR "cannot|illegal|incompatible|wrong"
var v3 = S{ make(map[uint]int) }	// ERROR "cannot|illegal|incompatible|wrong"
