// errchk $G -e $D/$F.go

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type I interface { m() }
type T struct { m func() }
type M struct {}
func (M) m() {}

func main() {
	var t T
	var m M
	var i I
	
	i = m
	i = t	// ERROR "not a method|has no methods"
	_ = i
}
