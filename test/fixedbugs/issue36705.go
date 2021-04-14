// +build cgo
// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// #include <stdlib.h>
// #include <unistd.h>
import "C"

import "os"

func main() {
	os.Setenv("FOO", "bar")
	s := C.GoString(C.getenv(C.CString("FOO")))
	if s != "bar" {
		panic("bad setenv, environment variable only has value \"" + s + "\"")
	}
	os.Unsetenv("FOO")
	s = C.GoString(C.getenv(C.CString("FOO")))
	if s != "" {
		panic("bad unsetenv, environment variable still has value \"" + s + "\"")
	}
}
