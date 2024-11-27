// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f(...any)

func _(x int, s []int) {
	f(0, x /* ERROR "have (number, int...)\n\twant (...any)" */ ...)
	f(0, s /* ERROR "have (number, []int...)\n\twant (...any)" */ ...)
	f(0, 0 /* ERROR "have (number, number...)\n\twant (...any)" */ ...)
}
