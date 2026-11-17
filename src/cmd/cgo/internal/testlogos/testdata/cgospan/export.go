// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "C"

import "unsafe"

var exported unsafe.Pointer

//export span_export
func span_export() unsafe.Pointer {
	return exported
}
