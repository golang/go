// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// errchk $G $D/$F.go

package main

type T struct {
	v ();  // ERROR "field type"
}
