// compile

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Ensure that late expansion correctly handles an OpIData with type interface{}

package p

type S struct{}

func (S) M() {}

type I interface {
	M()
}

func f(i I) {
	o := i.(interface{})
	if _, ok := i.(*S); ok {
		o = nil
	}
	println(o)
}
