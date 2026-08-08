// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that a caller correctly restores its stack pointer even when
// the ABI0 NOFRAME assembly function it calls clobbers BP.
// See https://go.dev/issue/80710

package main

//go:noescape
func noframe(*int)

//go:noinline
func caller(indir func(*int)) {
	var useSomeStackSpace int
	noframe(&useSomeStackSpace)
	indir(nil)
}

func main() {
	caller(noframe)
}
