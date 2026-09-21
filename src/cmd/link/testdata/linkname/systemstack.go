// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Linkname systemstack is not allowed, even if it is
// defined in assembly.

package main

import _ "unsafe"

func f() {}

func main() {
	systemstack(f)
}

//go:linkname systemstack runtime.systemstack
func systemstack(func())
