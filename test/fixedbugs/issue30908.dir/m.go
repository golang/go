// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"

	"./b"
)

func main() {
	seed := "some things are better"
	bsl := []byte(seed)
	b.CallReadValues("/dev/null")
	vals, err := b.ReadValues(bsl)
	if vals["better"] != seed || err != nil {
		os.Exit(1)
	}
}
