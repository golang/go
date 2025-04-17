// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var decOps = []opData{}

var decBlocks = []blockData{}

func init() {
	archs = append(archs, arch{
		name:    "dec",
		ops:     decOps,
		blocks:  decBlocks,
		generic: true,
	})
}
