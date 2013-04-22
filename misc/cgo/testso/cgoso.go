// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgosotest

/*
#cgo LDFLAGS: -L. -lcgosotest
void init(void);
void sofunc(void);
*/
import "C"

func Test() {
	C.init()
	C.sofunc()
}

//export goCallback
func goCallback() {
}
