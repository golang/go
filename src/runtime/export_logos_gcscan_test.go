// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.logosarena

package runtime

import "unsafe"

// LogosArenaScanPointerBlock scans a block of unsafe.Pointer values. The caller
// must keep the block and every non-nil target in live loGo/s arena allocations.
// Such targets cannot enqueue heap work, so the benchmark needs no gcWork.
func LogosArenaScanPointerBlock(p unsafe.Pointer, size uintptr) {
	logosArenaScanObject(logosEfaceType(unsafe.Pointer(nil)), uintptr(p), size, nil, nil)
}
