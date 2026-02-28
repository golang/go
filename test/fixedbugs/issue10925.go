// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func prototype(xyz []string) {}
func main() {
	var got [][]string
	f := prototype
	f = func(ss []string) { got = append(got, ss) }
	for _, s := range []string{"one", "two", "three"} {
		f([]string{s})
	}
	if got[0][0] != "one" || got[1][0] != "two" || got[2][0] != "three" {
		// Bug's wrong output was [[three] [three] [three]]
		fmt.Println("Expected [[one] [two] [three]], got", got)
	}
}
