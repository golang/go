// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Linkname coroswitch is not allowed, even if iter.Pull
// is instantiated in the same package.

package main

import (
	"iter"
	"unsafe"
)

func seq(yield func(int) bool) {
	yield(123)
}

func main() {
	next, stop := iter.Pull(seq)
	next()
	stop()
	coroswitch(nil)
}

//go:linkname coroswitch runtime.coroswitch
func coroswitch(unsafe.Pointer)
