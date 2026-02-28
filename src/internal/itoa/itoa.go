// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Simple conversions to avoid depending on strconv.

package itoa

// Itoa converts val to a decimal string.
func Itoa(val int) string {
	if val < 0 {
		return "-" + Uitoa(uint(-val))
	}
	return Uitoa(uint(val))
}

// Uitoa converts val to a decimal string.
func Uitoa(val uint) string {
	if val == 0 { // avoid string allocation
		return "0"
	}
	var buf [20]byte // big enough for 64bit value base 10
	i := len(buf) - 1
	for val >= 10 {
		q := val / 10
		buf[i] = byte('0' + val - q*10)
		i--
		val = q
	}
	// val < 10
	buf[i] = byte('0' + val)
	return string(buf[i:])
}
