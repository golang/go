// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 52846: gofrontend crashed with alias as map key type

package p

type S struct {
	F string
}

type A = S

var M = map[A]int{A{""}: 0}
