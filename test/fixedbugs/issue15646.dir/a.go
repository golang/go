// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type T struct{}

func (T) m() string {
	return "m"
}

func (*T) mp() string {
	return "mp"
}

func F() func(T) string {
	return T.m // method expression
}

func Fp() func(*T) string {
	return (*T).mp // method expression
}
