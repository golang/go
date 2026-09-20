// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "reflect"

type E struct{}

func (E) M() {}

func g(x struct{ a, b int }) {
	reflect.ValueOf(x)
}

func gEmbedded(x struct {
	E
	b int
}) {
	reflect.ValueOf(x)
}

func f[T any](i interface{}) {
	switch i.(type) {
	case T:
	case struct{ a, b T }:
	}
}

func fEmbedded[T any](i interface{}) {
	switch i.(type) {
	case T:
	case struct {
		E
		b T
	}:
	}
}

func main() {
	f[int](0)
	f[any](0)
	fEmbedded[int](0)
	fEmbedded[any](0)
}
