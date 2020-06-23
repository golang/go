// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type it struct {
	Floats bool
	inner  string
}

func main() {
	i1 := it{Floats: true}
	if i1.floats { // ERROR "(type it .* field or method floats, but does have Floats)"
	}
	i2 := &it{floats: false} // ERROR "(but does have Floats)"
	_ = &it{InneR: "foo"}    // ERROR "(but does have inner)"
}
