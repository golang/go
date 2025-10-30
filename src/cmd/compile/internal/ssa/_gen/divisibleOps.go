// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var divisibleOps = []opData{}

var divisibleBlocks = []blockData{}

func init() {
	archs = append(archs, arch{
		name:    "divisible",
		ops:     divisibleOps,
		blocks:  divisibleBlocks,
		generic: true,
	})
}
