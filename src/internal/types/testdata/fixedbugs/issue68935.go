// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type A = struct {
	F string
	G int
}

func Make[T ~A]() T {
	return T{
		F: "blah",
		G: 1234,
	}
}

type N struct {
	F string
	G int
}

func _() {
	_ = Make[N]()
}
