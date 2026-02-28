// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// gccgo mishandled passing a struct with an empty field through
// reflect.Value.Call.

package main

import (
	"reflect"
)

type Empty struct {
	f1, f2 *byte
	empty struct{}
}

func F(e Empty, s []string) {
	if len(s) != 1 || s[0] != "hi" {
		panic("bad slice")
	}
}

func main() {
	reflect.ValueOf(F).Call([]reflect.Value{
		reflect.ValueOf(Empty{}),
		reflect.ValueOf([]string{"hi"}),
	})
}
