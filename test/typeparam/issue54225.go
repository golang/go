// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	One[TextValue]()
}

func One[V Value]() { Two[Node[V]]() }

func Two[V interface{ contentLen() int }]() {
	var v V
	v.contentLen()
}

type Value interface {
	Len() int
}

type Node[V Value] struct{}

func (Node[V]) contentLen() int {
	var value V
	return value.Len()
}

type TextValue struct{}

func (TextValue) Len() int { return 0 }
