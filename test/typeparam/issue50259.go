// errorcheck -G=3

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var x T[B]

type T[_ any] struct{}
type A T[B] // ERROR "invalid use of type alias B in recursive type"
type B = T[A]
