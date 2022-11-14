// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type T1 struct {
	*T2
}

type T2 struct {
}

func (t2 *T2) M() {
}

func F() {
	f(T1.M)
}

func f(f func(T1)) {
}
