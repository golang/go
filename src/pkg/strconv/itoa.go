// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

// Uitob64 returns the string representation of i in the given base.
func Uitob64(u uint64, base uint) string {
	if base < 2 || 36 < base {
		panic("invalid base " + Uitoa(base))
	}
	if u == 0 {
		return "0"
	}

	// Assemble decimal in reverse order.
	var buf [64]byte
	j := len(buf)
	b := uint64(base)
	for u > 0 {
		j--
		buf[j] = "0123456789abcdefghijklmnopqrstuvwxyz"[u%b]
		u /= b
	}

	return string(buf[j:])
}

// Itob64 returns the string representation of i in the given base.
func Itob64(i int64, base uint) string {
	if i == 0 {
		return "0"
	}

	if i < 0 {
		return "-" + Uitob64(-uint64(i), base)
	}
	return Uitob64(uint64(i), base)
}

// Itoa64 returns the decimal string representation of i.
func Itoa64(i int64) string { return Itob64(i, 10) }

// Uitoa64 returns the decimal string representation of i.
func Uitoa64(i uint64) string { return Uitob64(i, 10) }

// Uitob returns the string representation of i in the given base.
func Uitob(i uint, base uint) string { return Uitob64(uint64(i), base) }

// Itob returns the string representation of i in the given base.
func Itob(i int, base uint) string { return Itob64(int64(i), base) }

// Itoa returns the decimal string representation of i.
func Itoa(i int) string { return Itob64(int64(i), 10) }

// Uitoa returns the decimal string representation of i.
func Uitoa(i uint) string { return Uitob64(uint64(i), 10) }
