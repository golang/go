// errorcheck

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var (
	_ [10]int
	_ [10.0]int
	_ [float64(10)]int                // ERROR "invalid array bound|must be integer"
	_ [10 + 0i]int
	_ [complex(10, 0)]int
	_ [complex128(complex(10, 0))]int // ERROR "invalid array bound|must be integer"
	_ ['a']int
	_ [rune(65)]int
)
