// run

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 5244: the init order computation uses the wrong
// order for top-level blank identifier assignments.
// The example used to panic because it tries calling a
// nil function instead of assigning to f before.

package main

var f = func() int { return 1 }
var _ = f() + g()
var g = func() int { return 2 }

func main() {}
