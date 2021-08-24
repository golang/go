// run -goexperiment fieldtrack

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var i interface{} = new(T)
	if _, ok := i.(interface{ Bad() }); ok {
		panic("FAIL")
	}
}

type T struct{ U }

type U struct{}

//go:nointerface
func (*U) Bad() {}
