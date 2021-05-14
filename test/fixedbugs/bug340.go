// errorcheck

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 1606.

package main

func main() {
	var x interface{}
	switch t := x.(type) {
	case 0:		// ERROR "type"
		t.x = 1
		x.x = 1 // ERROR "type interface \{\}|reference to undefined field or method|interface with no methods|undefined"
	}
}
