// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "reflect"

type C struct {
}

func F(c *C) *C {
	return nil
}

func G() bool {
	var c *C
	return reflect.TypeOf(F).Out(0) == reflect.TypeOf(c)
}
