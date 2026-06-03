// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

// sizeVarint returns the size of the variable-length integer encoding of f.
// Copied from internal/quic/quicwire to break dependency that makes bundling
// into std more complicated.
func sizeVarint(v uint64) int {
	switch {
	case v <= 63:
		return 1
	case v <= 16383:
		return 2
	case v <= 1073741823:
		return 4
	case v <= 4611686018427387903:
		return 8
	default:
		panic("varint too large")
	}
}
