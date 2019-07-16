// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file

package p

type T struct{ x int }
type S struct{}

func (p *S) get() {
}

type I interface {
	get()
}

func F(i I) {
	i.get()
}
