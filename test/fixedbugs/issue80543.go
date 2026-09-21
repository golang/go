// compile -d=ssa/check/on

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type S struct {
	i int
	a [1]struct{}
}

func foo(f func(S)) {
	defer f(func() S {
		return S{0, [1]struct{}{}}
	}())
}
