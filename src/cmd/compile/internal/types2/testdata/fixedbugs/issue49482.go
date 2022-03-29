// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is tested when running "go test -run Manual"
// without source arguments. Use for one-off debugging.

package p

// The following is OK, per the special handling for type literals discussed in issue #49482.
type _[P *struct{}] struct{}
type _[P *int,] int
type _[P (*int),] int

const P = 2 // declare P to avoid noisy 'undeclared name' errors below.

// The following parse as invalid array types.
type _[P *int /* ERROR "int \(type\) is not an expression" */ ] int
type _[P /* ERROR non-function P */ (*int)] int

// The following should be parsed as a generic type, but is instead parsed as an array type.
type _[P *struct /* ERROR "not an expression" */ {}| int /* ERROR "not an expression" */ ] struct{}

// The following fails to parse, due to the '~'
type _[P *struct /* ERROR "not an expression" */ {}|~ /* ERROR "unexpected ~" */ int] struct{}
