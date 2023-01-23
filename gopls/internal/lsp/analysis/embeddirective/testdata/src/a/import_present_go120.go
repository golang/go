// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.20
// +build go1.20

package a

var (
	// Okay directive wise but the compiler will complain that
	// imports must appear before other declarations.
	//go:embed embedText // ok
	"foo"
)

import (
	"fmt"

	_ "embed"
)

// This is main function
func main() {
	fmt.Println(s)
}
