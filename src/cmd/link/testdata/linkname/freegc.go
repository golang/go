// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Linkname runtime.freegc is not allowed.

package main

import (
	_ "unsafe"
)

//go:linkname freegc runtime.freegc
func freegc()

func main() {
	freegc()
}
