// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"./a"
)

var v = a.S{}

func main() {
	want := "{{ 0}}"
	if got := fmt.Sprint(v.F); got != want {
		panic(got)
	}
}
