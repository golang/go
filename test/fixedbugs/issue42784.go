// compile

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Ensure that late expansion correctly set OpLoad argument type interface{}

package p

type iface interface {
	m()
}

type it interface{}

type makeIface func() iface

func f() {
	var im makeIface
	e := im().(it)
	g(e)
}

//go:noinline
func g(i it) {}
