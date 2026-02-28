// runoutput

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 7316
// This test exercises all types of numeric conversions, which was one
// of the sources of etype mismatch during register allocation in 8g.

package main

import "fmt"

const tpl = `
func init() {
	var i %s
	j := %s(i)
	_ = %s(j)
}
`

func main() {
	fmt.Println("package main")
	ntypes := []string{
		"byte", "rune", "uintptr",
		"float32", "float64",
		"int", "int8", "int16", "int32", "int64",
		"uint", "uint8", "uint16", "uint32", "uint64",
	}
	for i, from := range ntypes {
		for _, to := range ntypes[i:] {
			fmt.Printf(tpl, from, to, from)
		}
	}
	fmt.Println("func main() {}")
}
