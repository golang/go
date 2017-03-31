// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type T struct{ _ int }
func (t T) M() {}

type I interface { M() }

func F() {
	var t I = &T{}
	t.M() // call to the wrapper (*T).M
}
