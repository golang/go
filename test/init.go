// errorcheck

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that erroneous use of init is detected.
// Does not compile.

package main

func init() {
}

func main() {
	init()         // ERROR "undefined.*init"
	runtime.init() // ERROR "undefined.*runtime\.init"
	var _ = init   // ERROR "undefined.*init"
}
