// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && !arm64

package atomic

// Cas128 atomically compares the 16 bytes at *ptr to (old1, old2) and,
// if equal, replaces them with (new1, new2). On architectures without a
// native 128-bit atomic instruction, this delegates to the lock-table
// fallback in atomic_cas128_native.go. ptr must be 16-byte aligned.
//
//go:nosplit
func Cas128(ptr *[2]uint64, old1, old2, new1, new2 uint64) bool {
	return goCas128(ptr, old1, old2, new1, new2)
}
