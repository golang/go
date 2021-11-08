// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type A interface {
	M(i interface{}) interface{}
}

var a1 A
var a2 A

func V(p A, k, v interface{}) A {
	defer func() { a1, a2 = a2, a1 }()
	return a1
}
