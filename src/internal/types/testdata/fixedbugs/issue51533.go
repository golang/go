// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _(x any) {
	switch x {
	case 0:
		fallthrough // ERROR fallthrough statement out of place
		_ = x
	default:
	}

	switch x.(type) {
	case int:
		fallthrough // ERROR cannot fallthrough in type switch
	default:
	}
}
