// run -race -gcflags=all=-d=checkptr=0

//go:build ((linux && amd64) || (linux && ppc64le) || (darwin && amd64) || (freebsd && amd64) || (netbsd && amd64) || (windows && amd64)) && cgo

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Although -race turns on -d=checkptr, the explicit -d=checkptr=0
// should override it.

package main

import "unsafe"

var v1 = new([2]int16)
var v2 *[3]int64

func main() {
	v2 = (*[3]int64)(unsafe.Pointer(uintptr(unsafe.Pointer(&(*v1)[0]))))
}
