// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "sync"

type T struct {
	int;
	sync.Mutex;
}

func main() {
	{
		var x, y sync.Mutex;
		x = y;	// ERROR "assignment.*Mutex"
		_ = x;
	}
	{
		var x, y T;
		x = y;	// ERROR "assignment.*Mutex"
		_ = x;
	}
	{
		var x, y [2]sync.Mutex;
		x = y;	// ERROR "assignment.*Mutex"
		_ = x;
	}
	{
		var x, y [2]T;
		x = y;	// ERROR "assignment.*Mutex"
		_ = x;
	}
}
