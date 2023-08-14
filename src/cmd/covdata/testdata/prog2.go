// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"prog/dep"
)

//go:noinline
func fifth() {
	println("hubba")
}

//go:noinline
func sixth() {
	println("wha?")
}

func main() {
	println(dep.Dep1())
	if len(os.Args) > 1 {
		fifth()
	} else {
		sixth()
	}
}
