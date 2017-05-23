// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build gccgo

package cgotest

//typedef void (*PFunc)();
//PFunc success_cb;
import "C"

//export Test
func Test() {
	_ = C.success_cb
}
