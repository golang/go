// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct {
	GlobalName string
}

var t = T{Name: "foo"} // ERROR "unknown field 'Name' in struct literal of type T"

func (t T) Name() string {
	return t.GlobalName
}
