// +build arm

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure we can compile assembly with DIV and MOD in it.
// They get rewritten to runtime calls on GOARM=5.

package main

func f(x, y uint32)

func main() {
	f(5, 8)
}
