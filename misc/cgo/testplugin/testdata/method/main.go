// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 42579: methods of symbols exported from plugin must be live.

package main

import (
	"plugin"
	"reflect"
)

func main() {
	p, err := plugin.Open("plugin.so")
	if err != nil {
		panic(err)
	}

	x, err := p.Lookup("X")
	if err != nil {
		panic(err)
	}

	reflect.ValueOf(x).Elem().MethodByName("M").Call(nil)
}
