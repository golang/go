// compile -G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Foo[T any] struct {
        Val T
}

func (f Foo[T]) Bat() {}

type Bar struct {
        Foo[int]
}

func foo() {
        var b Bar
        b.Bat()
}
