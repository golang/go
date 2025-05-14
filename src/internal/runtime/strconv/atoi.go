// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

import (
	"internal/runtime/math"
)

// Atoi64 parses an int64 from a string s.
// The bool result reports whether s is a number
// representable by a value of type int64.
func Atoi64(s string) (int64, bool) {
	if s == "" {
		return 0, false
	}

	neg := false
	if s[0] == '-' {
		neg = true
		s = s[1:]
	}

	un := uint64(0)
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c < '0' || c > '9' {
			return 0, false
		}
		if un > math.MaxUint64/10 {
			// overflow
			return 0, false
		}
		un *= 10
		un1 := un + uint64(c) - '0'
		if un1 < un {
			// overflow
			return 0, false
		}
		un = un1
	}

	if !neg && un > uint64(math.MaxInt64) {
		return 0, false
	}
	if neg && un > uint64(math.MaxInt64)+1 {
		return 0, false
	}

	n := int64(un)
	if neg {
		n = -n
	}

	return n, true
}

// Atoi is like Atoi64 but for integers
// that fit into an int.
func Atoi(s string) (int, bool) {
	if n, ok := Atoi64(s); n == int64(int(n)) {
		return int(n), ok
	}
	return 0, false
}

// Atoi32 is like Atoi but for integers
// that fit into an int32.
func Atoi32(s string) (int32, bool) {
	if n, ok := Atoi64(s); n == int64(int32(n)) {
		return int32(n), ok
	}
	return 0, false
}

