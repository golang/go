// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T[P any] struct{}

func (T[_]) A() {}

var _ = (T[int]).A
var _ = (*T[int]).A

var _ = (T /* ERROR cannot use generic type */).A
var _ = (*T /* ERROR cannot use generic type */).A
