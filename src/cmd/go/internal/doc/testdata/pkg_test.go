// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkg

import "fmt"

// Package example.
func Example() {
	fmt.Println("Package example output")
	// Output: Package example output
}

// Function example.
func ExampleExportedFunc() {
	fmt.Println("Function example output")
	// Output: Function example output
}

// Function example two.
func ExampleExportedFunc_two() {
	fmt.Println("Function example two output")
	// Output: Function example two output
}

// Type example.
func ExampleExportedType() {
	fmt.Println("Type example output")
	// Output: Type example output
}

// Method example.
func ExampleExportedType_ExportedMethod() {
	fmt.Println("Method example output")
	// Output: Method example output
}

// Multiline example.
func Example_multiline() {
	fmt.Println("Multiline\nexample\noutput")
	// Output:
	// Multiline
	// example
	// output
}

