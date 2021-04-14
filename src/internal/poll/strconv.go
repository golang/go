// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build plan9

// Simple conversions to avoid depending on strconv.

package poll

// Convert integer to decimal string
func itoa(val int) string {
	if val < 0 {
		return "-" + uitoa(uint(-val))
	}
	return uitoa(uint(val))
}

// Convert unsigned integer to decimal string
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

// stringsHasSuffix is strings.HasSuffix. It reports whether s ends in
// suffix.
func stringsHasSuffix(s, suffix string) bool {
	return len(s) >= len(suffix) && s[len(s)-len(suffix):] == suffix
}
