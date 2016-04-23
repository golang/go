// run

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// PR65755: Incorrect type descriptor for type defined within method.

package main

import "reflect"

type S1 struct{}

func (S1) Fix() string {
	type s struct {
		f int
	}
	return reflect.TypeOf(s{}).Field(0).Name
}

type S2 struct{}

func (S2) Fix() string {
	type s struct {
		g bool
	}
	return reflect.TypeOf(s{}).Field(0).Name
}

func main() {
	f1 := S1{}.Fix()
	f2 := S2{}.Fix()
	if f1 != "f" || f2 != "g" {
		panic(f1 + f2)
	}
}
