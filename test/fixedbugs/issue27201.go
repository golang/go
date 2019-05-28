// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"strings"
)

func main() {
	f(nil)
}

func f(p *int32) {
	defer checkstack()
	v := *p         // panic should happen here, line 20
	sink = int64(v) // not here, line 21
}

var sink int64

func checkstack() {
	_ = recover()
	var buf [1024]byte
	n := runtime.Stack(buf[:], false)
	s := string(buf[:n])
	if strings.Contains(s, "issue27201.go:21 ") {
		panic("panic at wrong location")
	}
	if !strings.Contains(s, "issue27201.go:20 ") {
		panic("no panic at correct location")
	}
}
