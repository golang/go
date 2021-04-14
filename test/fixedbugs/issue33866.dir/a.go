// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type Builder struct {
	x int
}

func (tb Builder) Build() (out struct {
	x interface{}
	s string
}) {
	out.x = nil
	out.s = "hello!"
	return
}
