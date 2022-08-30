// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fillstruct

type emptyStruct[A any] struct{}

var _ = emptyStruct[int]{}

type basicStruct[T any] struct {
	foo T
}

var _ = basicStruct[int]{} // want ""

type twoArgStruct[F, B any] struct {
	foo F
	bar B
}

var _ = twoArgStruct[string, int]{} // want ""

var _ = twoArgStruct[int, string]{ // want ""
	bar: "bar",
}

type nestedStruct struct {
	bar   string
	basic basicStruct[int]
}

var _ = nestedStruct{}

func _[T any]() {
	type S struct{ t T }
	x := S{}
	_ = x
}
