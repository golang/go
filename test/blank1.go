// errchk $G -e $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package _	// ERROR "invalid package name _"

func main() {
	_()	// ERROR "cannot use _ as value"
	x := _+1	// ERROR "cannot use _ as value"
	_ = x
}
