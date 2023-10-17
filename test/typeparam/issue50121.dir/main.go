// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./a"
)

//go:noinline
func BuildInt() int {
	return a.BuildInt()
}

func main() {
	BuildInt()
}
