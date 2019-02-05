// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func F() { // ERROR "can inline F"
	print(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	print(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	print(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	print(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	print(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	print(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
}

func G() {
	F() // ERROR "inlining call to F"
	print(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	print(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	print(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	print(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	print(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	print(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
}
