// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9
// +build !plan9

package main

// void start(void);
import "C"

func init() {
	register("CgoExternalThreadPanic", CgoExternalThreadPanic)
}

func CgoExternalThreadPanic() {
	C.start()
	select {}
}

//export gopanic
func gopanic() {
	panic("BOOM")
}
