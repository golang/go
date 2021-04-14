// compile

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 6247: 5g used to be confused by the numbering
// of floating-point registers.

package main

var p map[string]interface{}
var v interface{}

func F() {
	p["hello"] = v.(complex128) * v.(complex128)
}
