// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	register(printerconfigFix)
}

var printerconfigFix = fix{
	name: "printerconfig",
	date: "2012-12-11",
	f:    noop,
	desc: `Add element keys to Config composite literals (removed).`,
}
