// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/ssa/ssacore"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
)

// zcse does an initial pass of common-subexpression elimination on the
// function for values with zero arguments to allow the more expensive cse
// to begin with a reduced number of values. Values are just relinked,
// nothing is deleted. A subsequent deadcode pass is required to actually
// remove duplicate expressions.
func zcse(f *ssacore.Func) {
	vals := make(map[vkey]*ssacore.Value)

	for _, b := range f.Blocks {
		for i := 0; i < len(b.Values); i++ {
			v := b.Values[i]
			if ssaop.OpcodeTable[v.Op].ArgLen == 0 {
				key := vkey{v.Op, keyFor(v), v.Aux, v.Type}
				if vals[key] == nil {
					vals[key] = v
					if b != f.Entry {
						// Move v to the entry block so it will dominate every block
						// where we might use it. This prevents the need for any dominator
						// calculations in this pass.
						v.Block = f.Entry
						f.Entry.Values = append(f.Entry.Values, v)
						last := len(b.Values) - 1
						b.Values[i] = b.Values[last]
						b.Values[last] = nil
						b.Values = b.Values[:last]

						i-- // process b.Values[i] again
					}
				}
			}
		}
	}

	for _, b := range f.Blocks {
		for _, v := range b.Values {
			for i, a := range v.Args {
				if ssaop.OpcodeTable[a.Op].ArgLen == 0 {
					key := vkey{a.Op, keyFor(a), a.Aux, a.Type}
					if rv, ok := vals[key]; ok {
						v.SetArg(i, rv)
					}
				}
			}
		}
	}
}

// vkey is a type used to uniquely identify a zero arg value.
type vkey struct {
	op ssaop.Op
	ai int64       // aux int
	ax ssacore.Aux // aux
	t  *types.Type // type
}

// keyFor returns the AuxInt portion of a  key structure uniquely identifying a
// zero arg value for the supported ops.
func keyFor(v *ssacore.Value) int64 {
	switch v.Op {
	case ssaop.OpConst64, ssaop.OpConst64F, ssaop.OpConst32F:
		return v.AuxInt
	case ssaop.OpConst32:
		return int64(int32(v.AuxInt))
	case ssaop.OpConst16:
		return int64(int16(v.AuxInt))
	case ssaop.OpConst8, ssaop.OpConstBool:
		return int64(int8(v.AuxInt))
	default:
		return v.AuxInt
	}
}
