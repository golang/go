// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue31540

type Y struct {
	q int
}

type Z map[int]int

type X = map[Y]Z

type A1 = X

type A2 = A1

type S struct {
	b int
	A2
}

func Hallo() S {
	return S{}
}
