// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type X int

func (x X) M() X { return x }

func F[T interface{ M() U }, U interface{ M() T }]() {}
func G()                                             { F[X, X]() }
