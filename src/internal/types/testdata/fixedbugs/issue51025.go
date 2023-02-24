// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var _ interface{ m() } = struct /* ERROR "m is a field, not a method" */ {
	m func()
}{}

var _ interface{ m() } = & /* ERROR "m is a field, not a method" */ struct {
	m func()
}{}

var _ interface{ M() } = struct /* ERROR "missing method M" */ {
	m func()
}{}

var _ interface{ M() } = & /* ERROR "missing method M" */ struct {
	m func()
}{}

// test case from issue
type I interface{ m() }
type T struct{ m func() }
type M struct{}

func (M) m() {}

func _() {
	var t T
	var m M
	var i I

	i = m
	i = t // ERROR "m is a field, not a method"
	_ = i
}
