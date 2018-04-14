// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86

// argListMax specifies upper arg count limit expected to be carried by obj.Prog.
// Max len(obj.Prog.RestArgs) can be inferred from this to be 4.
const argListMax int = 6

type argList [argListMax]uint8

type ytab struct {
	zcase   uint8
	zoffset uint8

	// Last arg is usually destination.
	// For unary instructions unaryDst is used to determine
	// if single argument is a source or destination.
	args argList
}

// Returns true if yt is compatible with args.
//
// Elements from args and yt.args are used to
// to index ycover table like `ycover[args[i]+yt.args[i]]`.
// This means that args should contain values that already
// multiplied by Ymax.
func (yt *ytab) match(args []int) bool {
	// Trailing Yxxx check is required to avoid a case
	// where shorter arg list is matched.
	// If we had exact yt.args length, it could be `yt.argc != len(args)`.
	if len(args) < len(yt.args) && yt.args[len(args)] != Yxxx {
		return false
	}

	for i := range args {
		if ycover[args[i]+int(yt.args[i])] == 0 {
			return false
		}
	}

	return true
}
