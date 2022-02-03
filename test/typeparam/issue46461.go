// errorcheck -G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T[U interface{ M() T[U] }] int // ERROR "invalid recursive type T"

type X int

func (X) M() T[X] { return 0 }
