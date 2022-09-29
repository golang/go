// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fillstruct

type emptyStruct[A any] struct{}

var _ = emptyStruct[int]{}

type basicStruct[T any] struct {
	foo T
}

var _ = basicStruct[int]{} // want `Fill basicStruct\[int\]`

type twoArgStruct[F, B any] struct {
	foo F
	bar B
}

var _ = twoArgStruct[string, int]{} // want `Fill twoArgStruct\[string, int\]`

var _ = twoArgStruct[int, string]{ // want `Fill twoArgStruct\[int, string\]`
	bar: "bar",
}

type nestedStruct struct {
	bar   string
	basic basicStruct[int]
}

var _ = nestedStruct{} // want "Fill nestedStruct"

func _[T any]() {
	type S struct{ t T }
	x := S{} // want "Fill S"
	_ = x
}

func Test() {
	var tests = []struct {
		a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p string
	}{
		{}, // want "Fill anonymous struct { a: string, b: string, c: string, ... }"
	}
	for _, test := range tests {
		_ = test
	}
}
