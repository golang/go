// compile

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T struct { // big enough to be an unSSAable type
	a, b, c, d, e, f int
}

func f(x interface{}, p *int) {
	_ = *p // trigger nil check here, removing it from below
	switch x := x.(type) {
	case *T:
		// Zero twice, so one of them will be removed by the deadstore pass
		*x = T{}
		*p = 0 // store op to prevent Zero ops from being optimized by the earlier opt pass rewrite rules
		*x = T{}
	}
}
