// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

func F1() {
L:
	goto L
}

func F2() {
L:
	for {
		break L
	}
}

func F3() {
L:
	for {
		continue L
	}
}

func F4() {
	switch {
	case true:
		fallthrough
	default:
	}
}

type T struct{}

func (T) M1() {
L:
	goto L
}

func (T) M2() {
L:
	for {
		break L
	}
}

func (T) M3() {
L:
	for {
		continue L
	}
}

func (T) M4() {
	switch {
	case true:
		fallthrough
	default:
	}
}
