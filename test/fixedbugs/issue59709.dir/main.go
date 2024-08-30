// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./dcache"
)

func main() {
	var m dcache.Module
	m.Configure("x")
	m.Configure("y")
	var e error
	m.Blurb("x", e)
}
