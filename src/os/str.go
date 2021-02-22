// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Simple conversions to avoid depending on strconv.

package os

// itoa converts val (an int) to a decimal string.
func itoa(val int) string {
	if val < 0 {
		return "-" + uitoa(uint(-val))
	}
	return uitoa(uint(val))
}

// uitoa converts val (a uint) to a decimal string.
func uitoa(val uint) string {
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

// itox converts val (an int) to a hexdecimal string.
func itox(val int) string {
	if val < 0 {
		return "-" + uitox(uint(-val))
	}
	return uitox(uint(val))
}

const hex = "0123456789abcdef"

// uitox converts val (a uint) to a hexdecimal string.
func uitox(val uint) string {
	if val == 0 { // avoid string allocation
		return "0x0"
	}
	var buf [20]byte // big enough for 64bit value base 16 + 0x
	i := len(buf) - 1
	for val >= 16 {
		q := val / 16
		buf[i] = hex[val%16]
		i--
		val = q
	}
	// val < 16
	buf[i] = hex[val%16]
	i--
	buf[i] = 'x'
	i--
	buf[i] = '0'
	return string(buf[i:])
}
