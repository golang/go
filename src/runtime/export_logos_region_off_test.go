// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.logosarena

package runtime

import "unsafe"

func LogosRegionOpen() (h unsafe.Pointer, gen uintptr) {
	return logosRegionOpen()
}

func LogosRegionBytes() int {
	return logosRegionBytes()
}
