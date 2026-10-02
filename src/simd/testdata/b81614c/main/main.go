// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"b81614/lib"
	"simd"
)

func main() {
	var blank func() simd.Uint8s = lib.F
	_ = blank
}
