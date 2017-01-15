// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var (
	e interface{}
	s = struct{ a *int }{}
	b = e == s
)

func test(obj interface{}) {
	if obj != struct{ a *string }{} {
	}
}
