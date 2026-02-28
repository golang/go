// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type A struct {
	x int
}

type AI interface {
	bar()
}

type AC int

func (ab AC) bar() {
}

const (
	ACC = AC(101)
)

//go:noinline
func W(a A, k, v interface{}) A {
	return A{3}
}
