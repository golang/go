// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams

import (
	"bytes"
	"fmt"
	"net/http"
)

type parent[T any] interface {
	n(f T)
}

type yuh[T any] struct {
	a T
}

func (y *yuh[int]) n(f bool) {
	for i := 0; i < 10; i++ {
		fmt.Println(i)
	}
}

func a[T comparable](i1 int, i2 T, i3 int) int { // want "potentially unused parameter: 'i2'"
	i3 += i1
	_ = func(z int) int { // want "potentially unused parameter: 'z'"
		_ = 1
		return 1
	}
	return i3
}

func b[T any](c bytes.Buffer) { // want "potentially unused parameter: 'c'"
	_ = 1
}

func z[T http.ResponseWriter](h T, _ *http.Request) { // want "potentially unused parameter: 'h'"
	fmt.Println("Before")
}

func l(h http.Handler) http.Handler {
	return http.HandlerFunc(z[http.ResponseWriter])
}

func mult(a, b int) int { // want "potentially unused parameter: 'b'"
	a += 1
	return a
}

func y[T any](a T) {
	panic("yo")
}
