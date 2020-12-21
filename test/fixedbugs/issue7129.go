// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 7129: inconsistent "wrong arg type" error for multivalued g in f(g())

package main

func f(int) {}

func g() bool { return true }

func h(int, int) {}

func main() {
	f(g())        // ERROR "in argument to f|incompatible type|cannot convert"
	f(true)       // ERROR "in argument to f|incompatible type|cannot convert"
	h(true, true) // ERROR "in argument to h|incompatible type|cannot convert"
}
