// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check import package contains type alias in function
// with the same name with an export type not panic

package main

import (
	"fmt"

	"./a"
)

func main() {
	fmt.Println(a.T{})
	a.F()
}
