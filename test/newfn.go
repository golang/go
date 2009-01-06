// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// errchk $G $D/$F.go

package main

func main()
{
	f := new(());	// ERROR "new"
	g := new((x int, f float) string);	// ERROR "new"
	h := new(());	// ok
}
