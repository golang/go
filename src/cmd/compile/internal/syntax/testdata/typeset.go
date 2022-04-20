// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains test cases for typeset-only constraint elements.

package p

type (
        _[_ t] t
        _[_ ~t] t
        _[_ t|t] t
        _[_ ~t|t] t
        _[_ t|~t] t
        _[_ ~t|~t] t

        _[_ t, _, _ t|t] t
        _[_ t, _, _ ~t|t] t
        _[_ t, _, _ t|~t] t
        _[_ t, _, _ ~t|~t] t

        _[_ t.t] t
        _[_ ~t.t] t
        _[_ t.t|t.t] t
        _[_ ~t.t|t.t] t
        _[_ t.t|~t.t] t
        _[_ ~t.t|~t.t] t

        _[_ t, _, _ t.t|t.t] t
        _[_ t, _, _ ~t.t|t.t] t
        _[_ t, _, _ t.t|~t.t] t
        _[_ t, _, _ ~t.t|~t.t] t

        _[_ struct{}] t
        _[_ ~struct{}] t

        _[_ struct{}|t] t
        _[_ ~struct{}|t] t
        _[_ struct{}|~t] t
        _[_ ~struct{}|~t] t

        _[_ t|struct{}] t
        _[_ ~t|struct{}] t
        _[_ t|~struct{}] t
        _[_ ~t|~struct{}] t

        // test cases for issue #49175
        _[_ []t]t
        _[_ [1]t]t
        _[_ ~[]t]t
        _[_ ~[1]t]t
        t [ /* ERROR type parameters must be named */ t[0]]t
)

// test cases for issue #49174
func _[_ t]() {}
func _[_ []t]() {}
func _[_ [1]t]() {}
func _[_ []t | t]() {}
func _[_ [1]t | t]() {}
func _[_ t | []t]() {}
func _[_ []t | []t]() {}
func _[_ [1]t | [1]t]() {}
func _[_ t[t] | t[t]]() {}

// Single-expression type parameter lists and those that don't start
// with a (type parameter) name are considered array sizes.
// The term must be a valid expression (it could be a type - and then
// a type-checker will complain - but we don't allow ~ in the expr).
type (
        _[t] t
        _[/* ERROR unexpected ~ */ ~t] t
        _[t|t] t
        _[/* ERROR unexpected ~ */ ~t|t] t
        _[t| /* ERROR unexpected ~ */ ~t] t
        _[/* ERROR unexpected ~ */ ~t|~t] t
)

type (
        _[_ t, t /* ERROR missing type constraint */ ] t
        _[_ ~t, t /* ERROR missing type constraint */ ] t
        _[_ t, /* ERROR type parameters must be named */ ~t] t
        _[_ ~t, /* ERROR type parameters must be named */ ~t] t

        _[_ t|t, /* ERROR type parameters must be named */ t|t] t
        _[_ ~t|t, /* ERROR type parameters must be named */ t|t] t
        _[_ t|t, /* ERROR type parameters must be named */ ~t|t] t
        _[_ ~t|t, /* ERROR type parameters must be named */ ~t|t] t
)
