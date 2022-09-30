// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var dec64Ops = []opData{}

var dec64Blocks = []blockData{}

func init() {
	archs = append(archs, arch{
		name:    "dec64",
		ops:     dec64Ops,
		blocks:  dec64Blocks,
		generic: true,
	})
}
