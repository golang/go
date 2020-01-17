// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/go2go"
)

// translate writes .go files for all .go2 files in dir.
func translate(dir string) {
	if err := go2go.Rewrite(dir); err != nil {
		die(err.Error())
	}
}
