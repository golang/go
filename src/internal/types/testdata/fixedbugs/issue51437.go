// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T struct{}

func (T) m() []int { return nil }

func f(x T) {
	for _, x := range func() []int {
		return x.m() // x declared in parameter list of f
	}() {
		_ = x // x declared by range clause
	}
}
