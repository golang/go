// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

var decArgsOps = []opData{}

var decArgsBlocks = []blockData{}

func init() {
	archs = append(archs, arch{
		name:    "decArgs",
		ops:     decArgsOps,
		blocks:  decArgsBlocks,
		generic: true,
	})
}
