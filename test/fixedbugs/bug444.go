// run

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The no-op conversion here used to confuse the compiler
// into doing a load-effective-address of nil.
// See issue 3670.

package main

import "reflect"

type T interface {}

var x bool

func main() {
        reflect.TypeOf(nil)
        reflect.TypeOf(T(nil)) // used to miscompile
        shouldPanic()
}

func f() byte {
	return []byte(nil)[0] // used to miscompile
}

func shouldPanic() {
	defer func() {
		if recover() == nil {
			panic("not panicking")
		}
	}()
	f()
}
