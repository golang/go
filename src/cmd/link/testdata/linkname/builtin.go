// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Linkname builtin symbols (that is not already linknamed,
// e.g. mapaccess1) is not allowed.

package main

import "unsafe"

func main() {
	mapaccess1(nil, nil, nil)
}

//go:linkname mapaccess1 runtime.mapaccess1
func mapaccess1(t, m, k unsafe.Pointer) unsafe.Pointer
