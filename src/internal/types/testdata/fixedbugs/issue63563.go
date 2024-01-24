// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var (
	_ = int8(1 /* ERROR "constant 255 overflows int8" */ <<8 - 1)
	_ = int16(1 /* ERROR "constant 65535 overflows int16" */ <<16 - 1)
	_ = int32(1 /* ERROR "constant 4294967295 overflows int32" */ <<32 - 1)
	_ = int64(1 /* ERROR "constant 18446744073709551615 overflows int64" */ <<64 - 1)

	_ = uint8(1 /* ERROR "constant 256 overflows uint8" */ << 8)
	_ = uint16(1 /* ERROR "constant 65536 overflows uint16" */ << 16)
	_ = uint32(1 /* ERROR "constant 4294967296 overflows uint32" */ << 32)
	_ = uint64(1 /* ERROR "constant 18446744073709551616 overflows uint64" */ << 64)
)

func _[P int8 | uint8]() {
	_ = P(0)
	_ = P(1 /* ERROR "constant 255 overflows int8 (in P)" */ <<8 - 1)
}

func _[P int16 | uint16]() {
	_ = P(0)
	_ = P(1 /* ERROR "constant 65535 overflows int16 (in P)" */ <<16 - 1)
}

func _[P int32 | uint32]() {
	_ = P(0)
	_ = P(1 /* ERROR "constant 4294967295 overflows int32 (in P)" */ <<32 - 1)
}

func _[P int64 | uint64]() {
	_ = P(0)
	_ = P(1 /* ERROR "constant 18446744073709551615 overflows int64 (in P)" */ <<64 - 1)
}
