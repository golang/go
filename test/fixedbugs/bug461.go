// run

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// part two of issue 4124. Make sure reflect doesn't mark the field as exported.

package main

import "reflect"

var T struct {
	int
}

func main() {
	v := reflect.ValueOf(&T)
	v = v.Elem().Field(0)
	if v.CanSet() {
		panic("int should be unexported")
	}
}
