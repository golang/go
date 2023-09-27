// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type (
        // 0 and 1-element []-lists are syntactically valid
        _[A, B /* ERROR missing type constraint */ ] int
        _[A, /* ERROR type parameters must be named */ interface{}] int
        _[A, B, C /* ERROR missing type constraint */ ] int
        _[A B, C /* ERROR missing type constraint */ ] int
        _[A B, /* ERROR type parameters must be named */ interface{}] int
        _[A B, /* ERROR type parameters must be named */ interface{}, C D] int
        _[A B, /* ERROR type parameters must be named */ interface{}, C, D] int
        _[A B, /* ERROR type parameters must be named */ interface{}, C, interface{}] int
        _[A B, C interface{}, D, /* ERROR type parameters must be named */ interface{}] int
)

// function type parameters use the same parsing routine - just have a couple of tests

func _[A, B /* ERROR missing type constraint */ ]() {}
func _[A, /* ERROR type parameters must be named */ interface{}]() {}
