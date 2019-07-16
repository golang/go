// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type m struct {
	S string
}

var g = struct {
	m
	P string
}{
	m{"a"},
	"",
}

type S struct{}

func (s *S) M(p string) {
	r := g
	r.P = p
}
