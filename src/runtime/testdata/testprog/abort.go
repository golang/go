// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import _ "unsafe" // for go:linkname

func init() {
	register("Abort", Abort)
}

//go:linkname runtimeAbort runtime.abort
func runtimeAbort()

func Abort() {
	defer func() {
		recover()
		panic("BAD: recovered from abort")
	}()
	runtimeAbort()
	println("BAD: after abort")
}
