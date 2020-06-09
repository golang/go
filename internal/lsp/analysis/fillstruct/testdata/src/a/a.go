// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fillstruct

import (
	data "b"
)

type emptyStruct struct{}

var _ = emptyStruct{}

type basicStruct struct {
	foo int
}

var _ = basicStruct{} // want ""

type twoArgStruct struct {
	foo int
	bar string
}

var _ = twoArgStruct{} // want ""

var _ = twoArgStruct{
	bar: "bar",
}

type nestedStruct struct {
	bar   string
	basic basicStruct
}

var _ = nestedStruct{} // want ""

var _ = data.B{} // want ""
