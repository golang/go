// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 52841: gofrontend crashed writing export data

package p

func F() {
	x := ([17][1]interface {
		Method9()
		Method10()
	}{
		func() (V47 [1]interface {
			Method9()
			Method10()
		}) {
			return
		}(),
		func(V48 string) (V49 [1]interface {
			Method9()
			Method10()
		}) {
			return
		}("440"),
	})
	_ = x
}
