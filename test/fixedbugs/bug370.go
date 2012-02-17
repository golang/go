// run

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// issue 2337
// The program deadlocked.

import "runtime"

func main() {
	runtime.GOMAXPROCS(2)
	runtime.GC()
	runtime.GOMAXPROCS(1)
}
