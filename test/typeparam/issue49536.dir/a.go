// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

func F() interface{} { return new(T[int]) }

type T[P any] int

func (x *T[P]) One() int { return x.Two() }
func (x *T[P]) Two() int { return 0 }
