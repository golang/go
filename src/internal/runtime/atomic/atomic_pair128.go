// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package atomic

// load128 atomically loads the pair stored at *ptr and returns it.
// ptr must be 16-byte aligned.
func load128(ptr *[2]uint64) (lo, hi uint64) {
	for {
		lo = Load64(&ptr[0])
		hi = Load64(&ptr[1])
		if cas128(ptr, lo, hi, lo, hi) {
			return
		}
	}
}

// store128 atomically stores (lo, hi) into *ptr.
// ptr must be 16-byte aligned.
func store128(ptr *[2]uint64, lo, hi uint64) {
	for {
		old0 := Load64(&ptr[0])
		old1 := Load64(&ptr[1])
		if cas128(ptr, old0, old1, lo, hi) {
			return
		}
	}
}
