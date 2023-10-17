// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func F(s string) {
	switch s[0] {
	case 'a':
		case s[2] { // ERROR unexpected {
		case 'b':
		}
	}
} // ERROR non-declaration statement
