// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package maps_test

import (
	"fmt"
	"maps"
	"strings"
)

func ExampleDeleteFunc() {
	m := map[string]int{
		"one":   1,
		"two":   2,
		"three": 3,
		"four":  4,
	}
	maps.DeleteFunc(m, func(k string, v int) bool {
		return v%2 != 0 // delete odd values
	})
	fmt.Println(m)
	// Output:
	// map[four:4 two:2]
}

func ExampleEqualFunc() {
	m1 := map[int]string{
		1:    "one",
		10:   "Ten",
		1000: "THOUSAND",
	}
	m2 := map[int][]byte{
		1:    []byte("One"),
		10:   []byte("Ten"),
		1000: []byte("Thousand"),
	}
	eq := maps.EqualFunc(m1, m2, func(v1 string, v2 []byte) bool {
		return strings.ToLower(v1) == strings.ToLower(string(v2))
	})
	fmt.Println(eq)
	// Output:
	// true
}
