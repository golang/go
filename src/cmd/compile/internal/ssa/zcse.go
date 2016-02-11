// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// zcse does an initial pass of common-subexpression elimination on the
// function for values with zero arguments to allow the more expensive cse
// to begin with a reduced number of values. Values are just relinked,
// nothing is deleted. A subsequent deadcode pass is required to actually
// remove duplicate expressions.
func zcse(f *Func) {
	vals := make(map[vkey]*Value)

	for _, b := range f.Blocks {
		for i := 0; i < len(b.Values); {
			v := b.Values[i]
			next := true
			switch v.Op {
			case OpSB, OpConst64, OpConst32, OpConst16, OpConst8, OpConst64F,
				OpConst32F, OpConstBool, OpConstNil, OpConstSlice, OpConstInterface:
				key := vkey{v.Op, keyFor(v), typeStr(v)}
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

						// process b.Values[i] again
						next = false
					}
				}
			}
			if next {
				i++
			}
		}
	}

	for _, b := range f.Blocks {
		for _, v := range b.Values {
			for i, a := range v.Args {
				// TODO: encode arglen in the opcode table, then do this switch with a table lookup?
				switch a.Op {
				case OpSB, OpConst64, OpConst32, OpConst16, OpConst8, OpConst64F,
					OpConst32F, OpConstBool, OpConstNil, OpConstSlice, OpConstInterface:
					key := vkey{a.Op, keyFor(a), typeStr(a)}
					if rv, ok := vals[key]; ok {
						v.Args[i] = rv
					}
				}
			}
		}
	}
}

// vkey is a type used to uniquely identify a zero arg value.
type vkey struct {
	op Op
	a  int64  // aux
	t  string // type
}

// typeStr returns a string version of the type of v.
func typeStr(v *Value) string {
	if v.Type == nil {
		return ""
	}
	return v.Type.String()
}

// keyFor returns the AuxInt portion of a  key structure uniquely identifying a
// zero arg value for the supported ops.
func keyFor(v *Value) int64 {
	switch v.Op {
	case OpConst64, OpConst64F, OpConst32F:
		return v.AuxInt
	case OpConst32:
		return int64(int32(v.AuxInt))
	case OpConst16:
		return int64(int16(v.AuxInt))
	case OpConst8, OpConstBool:
		return int64(int8(v.AuxInt))
	default:
		// Also matches OpSB, OpConstNil, OpConstSlice, OpConstInterface:
		return 0
	}
}
