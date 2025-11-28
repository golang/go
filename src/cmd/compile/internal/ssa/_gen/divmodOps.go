// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var divmodOps = []opData{}

var divmodBlocks = []blockData{}

func init() {
	archs = append(archs, arch{
		name:    "divmod",
		ops:     divmodOps,
		blocks:  divmodBlocks,
		generic: true,
	})
}
