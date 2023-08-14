// build

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func main() {
	a, b := 5, 7
	fmt.Println(min(a, b))
	fmt.Println(max(a, b))
}
