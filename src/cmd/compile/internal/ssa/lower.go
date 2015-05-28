// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "log"

//go:generate go run rulegen/rulegen.go rulegen/lower_amd64.rules lowerBlockAMD64 lowerValueAMD64 lowerAmd64.go

// convert to machine-dependent ops
func lower(f *Func) {
	// repeat rewrites until we find no more rewrites
	applyRewrite(f, f.Config.lowerBlock, f.Config.lowerValue)

	// Check for unlowered opcodes, fail if we find one.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op < OpGenericEnd && v.Op != OpFP && v.Op != OpSP && v.Op != OpArg && v.Op != OpCopy && v.Op != OpPhi {
				log.Panicf("%s not lowered", v.LongString())
			}
		}
	}
}
