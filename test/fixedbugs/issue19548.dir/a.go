// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type Mode uint

func (m Mode) String() string { return "mode string" }
func (m *Mode) Addr() *Mode   { return m }

type Stringer interface {
	String() string
}

var global Stringer
var m Mode

func init() {
	// force compilation of the (*Mode).String() wrapper
	global = &m
}

func String() string {
	return global.String() + Mode(0).String()
}
