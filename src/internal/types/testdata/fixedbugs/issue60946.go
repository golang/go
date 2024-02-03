// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Tn interface{ m() }
type T1 struct{}
type T2 struct{}

func (*T1) m() {}
func (*T2) m() {}

func g[P any](...P) {}

func _() {
	var t interface{ m() }
	var tn Tn
	var t1 *T1
	var t2 *T2

	// these are ok (interface types only)
	g(t, t)
	g(t, tn)
	g(tn, t)
	g(tn, tn)

	// these are not ok (interface and non-interface types)
	g(t, t1 /* ERROR "does not match" */)
	g(t1, t /* ERROR "does not match" */)
	g(tn, t1 /* ERROR "does not match" */)
	g(t1, tn /* ERROR "does not match" */)

	g(t, t1 /* ERROR "does not match" */, t2)
	g(t1, t2 /* ERROR "does not match" */, t)
	g(tn, t1 /* ERROR "does not match" */, t2)
	g(t1, t2 /* ERROR "does not match" */, tn)
}
