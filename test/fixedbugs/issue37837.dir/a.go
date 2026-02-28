// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

func F(i interface{}) int { // ERROR "can inline F" "i does not escape"
	switch i.(type) {
	case nil:
		return 0
	case int:
		return 1
	case float64:
		return 2
	default:
		return 3
	}
}

func G(i interface{}) interface{} { // ERROR "can inline G" "leaking param: i"
	switch i := i.(type) {
	case nil: // ERROR "moved to heap: i"
		return &i
	case int: // ERROR "moved to heap: i"
		return &i
	case float64: // ERROR "moved to heap: i"
		return &i
	case string, []byte: // ERROR "moved to heap: i"
		return &i
	default: // ERROR "moved to heap: i"
		return &i
	}
}
