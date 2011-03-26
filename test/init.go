// errchk $G -e $D/$F.go

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime"

func init() {
}

func main() {
	init()         // ERROR "undefined.*init"
	runtime.init() // ERROR "unexported.*runtime\.init"
	var _ = init   // ERROR "undefined.*init"
}
