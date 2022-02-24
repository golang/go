// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

// Type I is the first basic test for the issue, which relates to a type that is recursive
// via a type constraint.  (In this test, I -> IConstraint -> MyStruct -> I.)
type JsonRaw []byte

type MyStruct struct {
	x *I[JsonRaw]
}

type IConstraint interface {
	JsonRaw | MyStruct
}

type I[T IConstraint] struct {
}
