// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T[U interface{ M() T /* ERROR "too many type arguments for type T" */ [U, int] }] int

type X int

func (X) M() T[X] { return 0 }
