// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// version 1
var x1 T1[B1]

type T1[_ any] struct{}
type A1 T1[B1]
type B1 = T1[A1]

// version 2
type T2[_ any] struct{}
type A2 T2[B2]
type B2 = T2[A2]

var x2 T2[B2]
