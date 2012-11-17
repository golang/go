// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package cgo contains runtime support for code generated
by the cgo tool.  See the documentation for the cgo command
for details on using cgo.
*/
package cgo

/*

#cgo darwin LDFLAGS: -lpthread
#cgo freebsd LDFLAGS: -lpthread
#cgo linux LDFLAGS: -lpthread
#cgo netbsd LDFLAGS: -lpthread
#cgo openbsd LDFLAGS: -lpthread
#cgo windows LDFLAGS: -lm -mthreads

#cgo CFLAGS: -Wall -Werror

*/
import "C"

// Supports _cgo_panic by converting a string constant to an empty
// interface.

func cgoStringToEface(s string, ret *interface{}) {
	*ret = s
}
