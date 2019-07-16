// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package one

type T1 int
type T2 []T1
type T3 T2

func F1(T2) {
}

func (p *T1) M1() T3 {
	return nil
}

func (p T3) M2() {
}
