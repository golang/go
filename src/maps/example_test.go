// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package maps_test

import (
	"fmt"
	"maps"
	"strings"
)

func ExampleClone() {
	m1 := map[string]int{
		"key": 1,
	}
	m2 := maps.Clone(m1)
	m2["key"] = 100
	fmt.Println(m1["key"])
	fmt.Println(m2["key"])

	m3 := map[string][]int{
		"key": {1, 2, 3},
	}
	m4 := maps.Clone(m3)
	fmt.Println(m4["key"][0])
	m4["key"][0] = 100
	fmt.Println(m3["key"][0])
	fmt.Println(m4["key"][0])

	// Output:
	// 1
	// 100
	// 1
	// 100
	// 100
}

func ExampleCopy() {
	m1 := map[string]int{
		"one": 1,
		"two": 2,
	}
	m2 := map[string]int{
		"one": 10,
	}

	maps.Copy(m2, m1)
	fmt.Println("m2 is:", m2)

	m2["one"] = 100
	fmt.Println("m1 is:", m1)
	fmt.Println("m2 is:", m2)

	m3 := map[string][]int{
		"one": {1, 2, 3},
		"two": {4, 5, 6},
	}
	m4 := map[string][]int{
		"one": {7, 8, 9},
	}

	maps.Copy(m4, m3)
	fmt.Println("m4 is:", m4)

	m4["one"][0] = 100
	fmt.Println("m3 is:", m3)
	fmt.Println("m4 is:", m4)

	// Output:
	// m2 is: map[one:1 two:2]
	// m1 is: map[one:1 two:2]
	// m2 is: map[one:100 two:2]
	// m4 is: map[one:[1 2 3] two:[4 5 6]]
	// m3 is: map[one:[100 2 3] two:[4 5 6]]
	// m4 is: map[one:[100 2 3] two:[4 5 6]]
}

func ExampleDeleteFunc() {
	m := map[string]int{
		"one":   1,
		"two":   2,
		"three": 3,
		"four":  4,
	}
	maps.DeleteFunc(m, func { k, v -> v%2 != 0 })// delete odd values

	fmt.Println(m)
	// Output:
	// map[four:4 two:2]
}

func ExampleEqual() {
	m1 := map[int]string{
		1:    "one",
		10:   "Ten",
		1000: "THOUSAND",
	}
	m2 := map[int]string{
		1:    "one",
		10:   "Ten",
		1000: "THOUSAND",
	}
	m3 := map[int]string{
		1:    "one",
		10:   "ten",
		1000: "thousand",
	}

	fmt.Println(maps.Equal(m1, m2))
	fmt.Println(maps.Equal(m1, m3))
	// Output:
	// true
	// false
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
	eq := maps.EqualFunc(m1, m2, func { v1, v2 -> strings.ToLower(v1) == strings.ToLower(string(v2)) })
	fmt.Println(eq)
	// Output:
	// true
}
