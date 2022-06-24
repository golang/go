// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type _ func /* ERROR function type must have no type parameters */ [ /* ERROR empty type parameter list */ ]()
type _ func /* ERROR function type must have no type parameters */ [ x /* ERROR missing type constraint */ ]()
type _ func /* ERROR function type must have no type parameters */ [P any]()

var _ = (func /* ERROR function type must have no type parameters */ [P any]())(nil)
var _ = func /* ERROR function type must have no type parameters */ [P any]() {}

type _ interface{
        m /* ERROR interface method must have no type parameters */ [P any]()
}
