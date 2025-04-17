// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that recover arguments requirements are enforced by the
// compiler.

package main

func main() {
	_ = recover()     // OK
	_ = recover(1)    // ERROR "too many arguments"
	_ = recover(1, 2) // ERROR "too many arguments"
}
