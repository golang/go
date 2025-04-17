// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import _ "unsafe"

type schedt struct{}

//go:linkname sched runtime.sched
var sched schedt

func main() {
	select {
	default:
		println("hello")
	}
}
