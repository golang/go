// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Mer interface { M() }

func F[T Mer](p *T) {
	p.M /* ERROR "p.M undefined" */ ()
}

type MyMer int

func (MyMer) M() {}

func _() {
	F(new(MyMer))
	F[Mer](nil)
}
