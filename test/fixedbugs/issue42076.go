// run

// Copyright 2020 The Go Authors. All rights reserved.  Use of this
// source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package main

import "reflect"

//go:notinheap
type NIH struct {
}

var x, y NIH

func main() {
	if reflect.DeepEqual(&x, &y) != true {
		panic("should report true")
	}
}
