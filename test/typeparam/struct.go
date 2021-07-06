// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

type _E[T any] struct {
	v T
}

type _S1 struct {
	_E[int]
	v string
}

type _Eint = _E[int]
type _Ebool = _E[bool]

type _S2 struct {
	_Eint
	_Ebool
	v string
}

type _S3 struct {
	*_E[int]
}

func main() {
	s1 := _S1{_Eint{2}, "foo"}
	if got, want := s1._E.v, 2; got != want {
                panic(fmt.Sprintf("got %d, want %d", got, want))
	}
	s2 := _S2{_Eint{3}, _Ebool{true}, "foo"}
	if got, want := s2._Eint.v, 3; got != want {
                panic(fmt.Sprintf("got %d, want %d", got, want))
	}
	var s3 _S3
	s3._E = &_Eint{4}
	if got, want := s3._E.v, 4; got != want {
                panic(fmt.Sprintf("got %d, want %d", got, want))
	}
}
