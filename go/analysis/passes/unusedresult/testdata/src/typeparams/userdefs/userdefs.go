// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
//go:build go1.18

package userdefs

func MustUse[T interface{ ~int }](v T) T {
	return v + 1
}

type SingleTypeParam[T any] struct {
	X T
}

func (_ *SingleTypeParam[T]) String() string {
	return "SingleTypeParam"
}

type MultiTypeParam[T any, U any] struct {
	X T
	Y U
}

func (_ *MultiTypeParam[T, U]) String() string {
	return "MultiTypeParam"
}