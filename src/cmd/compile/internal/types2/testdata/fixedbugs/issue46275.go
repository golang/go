// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue46275

type N[T any] struct {
        *N[T]
        t T
}

func (n *N[T]) Elem() T {
        return n.t
}

type I interface {
        Elem() string
}

func _() {
        var n1 *N[string]
        var _ I = n1
        type NS N[string]
        var n2 *NS
        var _ I = n2
}
