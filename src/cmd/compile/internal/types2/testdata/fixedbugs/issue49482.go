// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// The following is OK, per the special handling for type literals discussed in issue #49482.
type _[P *struct{}] struct{}
type _[P *int,] int
type _[P (*int),] int

const P = 2 // declare P to avoid noisy 'undeclared name' errors below.

// The following parse as invalid array types due to parsing ambiguitiues.
type _ [P *int /* ERROR "int \(type\) is not an expression" */ ]int
type _ [P /* ERROR non-function P */ (*int)]int

// Adding a trailing comma or an enclosing interface resolves the ambiguity.
type _[P *int,] int
type _[P (*int),] int
type _[P interface{*int}] int
type _[P interface{(*int)}] int

// The following parse correctly as valid generic types.
type _[P *struct{} | int] struct{}
type _[P *struct{} | ~int] struct{}
