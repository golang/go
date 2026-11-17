// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.logosarena

package runtime

import "unsafe"

// LogosArenaScanTypedBlock scans size bytes using the type of elem.
// The caller must own a live arena buffer with readable, initialized padding
// through the final pointer word. Every pointer slot must be nil or point into
// a live arena allocation, so scanning cannot enqueue heap work.
func LogosArenaScanTypedBlock(elem any, p unsafe.Pointer, size uintptr) int64 {
	return logosArenaScanObject(logosEfaceType(elem), uintptr(p), size, nil, nil)
}
