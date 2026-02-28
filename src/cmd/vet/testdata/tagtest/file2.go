// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !testtag
// +build !testtag

package main

import "fmt"

func main() {
	fmt.Printf("%s", 0)
}
