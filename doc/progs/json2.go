// cmpout

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
)

func InterfaceExample() {
	var i interface{}
	i = "a string"
	i = 2011
	i = 2.777

	// STOP OMIT

	r := i.(float64)
	fmt.Println("the circle's area", math.Pi*r*r)

	// STOP OMIT

	switch v := i.(type) {
	case int:
		fmt.Println("twice i is", v*2)
	case float64:
		fmt.Println("the reciprocal of i is", 1/v)
	case string:
		h := len(v) / 2
		fmt.Println("i swapped by halves is", v[h:]+v[:h])
	default:
		// i isn't one of the types above
	}

	// STOP OMIT
}

func main() {
	InterfaceExample()
}
