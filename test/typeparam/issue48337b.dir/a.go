// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type Container[T any] struct {
	X T
}

func NewContainer[T any](x T) *Container[T] {
	return &Container[T]{x}
}

type MetaContainer struct {
	C *Container[Value]
}

type Value struct{}

func NewMetaContainer() *MetaContainer {
	c := NewContainer(Value{})
	// c := &Container[Value]{Value{}} // <-- this works
	return &MetaContainer{c}
}
