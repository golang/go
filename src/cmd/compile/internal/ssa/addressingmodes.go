// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// addressingModes combines address calculations into memory operations
// that can perform complicated addressing modes.
func addressingModes(f *Func) {
	switch f.Config.arch {
	default:
		// Most architectures can't do this.
		return
	case "amd64", "386":
		// TODO: s390x?
	}

	var tmp []*Value
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if !combineFirst[v.Op] {
				continue
			}
			// All matched operations have the pointer in arg[0].
			// All results have the pointer in arg[0] and the index in arg[1].
			// *Except* for operations which update a register,
			// which are marked with resultInArg0. Those have
			// the pointer in arg[1], and the corresponding result op
			// has the pointer in arg[1] and the index in arg[2].
			ptrIndex := 0
			if opcodeTable[v.Op].resultInArg0 {
				ptrIndex = 1
			}
			p := v.Args[ptrIndex]
			c, ok := combine[[2]Op{v.Op, p.Op}]
			if !ok {
				continue
			}
			// See if we can combine the Aux/AuxInt values.
			switch [2]auxType{opcodeTable[v.Op].auxType, opcodeTable[p.Op].auxType} {
			case [2]auxType{auxSymOff, auxInt32}:
				// TODO: introduce auxSymOff32
				if !is32Bit(v.AuxInt + p.AuxInt) {
					continue
				}
				v.AuxInt += p.AuxInt
			case [2]auxType{auxSymOff, auxSymOff}:
				if v.Aux != nil && p.Aux != nil {
					continue
				}
				if !is32Bit(v.AuxInt + p.AuxInt) {
					continue
				}
				if p.Aux != nil {
					v.Aux = p.Aux
				}
				v.AuxInt += p.AuxInt
			case [2]auxType{auxSymValAndOff, auxInt32}:
				vo := ValAndOff(v.AuxInt)
				if !vo.canAdd(p.AuxInt) {
					continue
				}
				v.AuxInt = vo.add(p.AuxInt)
			case [2]auxType{auxSymValAndOff, auxSymOff}:
				vo := ValAndOff(v.AuxInt)
				if v.Aux != nil && p.Aux != nil {
					continue
				}
				if !vo.canAdd(p.AuxInt) {
					continue
				}
				if p.Aux != nil {
					v.Aux = p.Aux
				}
				v.AuxInt = vo.add(p.AuxInt)
			case [2]auxType{auxSymOff, auxNone}:
				// nothing to do
			case [2]auxType{auxSymValAndOff, auxNone}:
				// nothing to do
			default:
				f.Fatalf("unknown aux combining for %s and %s\n", v.Op, p.Op)
			}
			// Combine the operations.
			tmp = append(tmp[:0], v.Args[:ptrIndex]...)
			tmp = append(tmp, p.Args...)
			tmp = append(tmp, v.Args[ptrIndex+1:]...)
			v.resetArgs()
			v.Op = c
			v.AddArgs(tmp...)
		}
	}
}

// combineFirst contains ops which appear in combine as the
// first part of the key.
var combineFirst = map[Op]bool{}

func init() {
	for k := range combine {
		combineFirst[k[0]] = true
	}
}

// For each entry k, v in this map, if we have a value x with:
//   x.Op == k[0]
//   x.Args[0].Op == k[1]
// then we can set x.Op to v and set x.Args like this:
//   x.Args[0].Args + x.Args[1:]
// Additionally, the Aux/AuxInt from x.Args[0] is merged into x.
var combine = map[[2]Op]Op{
	// amd64
	[2]Op{OpAMD64MOVBload, OpAMD64ADDQ}:  OpAMD64MOVBloadidx1,
	[2]Op{OpAMD64MOVWload, OpAMD64ADDQ}:  OpAMD64MOVWloadidx1,
	[2]Op{OpAMD64MOVLload, OpAMD64ADDQ}:  OpAMD64MOVLloadidx1,
	[2]Op{OpAMD64MOVQload, OpAMD64ADDQ}:  OpAMD64MOVQloadidx1,
	[2]Op{OpAMD64MOVSSload, OpAMD64ADDQ}: OpAMD64MOVSSloadidx1,
	[2]Op{OpAMD64MOVSDload, OpAMD64ADDQ}: OpAMD64MOVSDloadidx1,

	[2]Op{OpAMD64MOVBstore, OpAMD64ADDQ}:  OpAMD64MOVBstoreidx1,
	[2]Op{OpAMD64MOVWstore, OpAMD64ADDQ}:  OpAMD64MOVWstoreidx1,
	[2]Op{OpAMD64MOVLstore, OpAMD64ADDQ}:  OpAMD64MOVLstoreidx1,
	[2]Op{OpAMD64MOVQstore, OpAMD64ADDQ}:  OpAMD64MOVQstoreidx1,
	[2]Op{OpAMD64MOVSSstore, OpAMD64ADDQ}: OpAMD64MOVSSstoreidx1,
	[2]Op{OpAMD64MOVSDstore, OpAMD64ADDQ}: OpAMD64MOVSDstoreidx1,

	[2]Op{OpAMD64MOVBstoreconst, OpAMD64ADDQ}: OpAMD64MOVBstoreconstidx1,
	[2]Op{OpAMD64MOVWstoreconst, OpAMD64ADDQ}: OpAMD64MOVWstoreconstidx1,
	[2]Op{OpAMD64MOVLstoreconst, OpAMD64ADDQ}: OpAMD64MOVLstoreconstidx1,
	[2]Op{OpAMD64MOVQstoreconst, OpAMD64ADDQ}: OpAMD64MOVQstoreconstidx1,

	[2]Op{OpAMD64MOVBload, OpAMD64LEAQ1}:  OpAMD64MOVBloadidx1,
	[2]Op{OpAMD64MOVWload, OpAMD64LEAQ1}:  OpAMD64MOVWloadidx1,
	[2]Op{OpAMD64MOVWload, OpAMD64LEAQ2}:  OpAMD64MOVWloadidx2,
	[2]Op{OpAMD64MOVLload, OpAMD64LEAQ1}:  OpAMD64MOVLloadidx1,
	[2]Op{OpAMD64MOVLload, OpAMD64LEAQ4}:  OpAMD64MOVLloadidx4,
	[2]Op{OpAMD64MOVLload, OpAMD64LEAQ8}:  OpAMD64MOVLloadidx8,
	[2]Op{OpAMD64MOVQload, OpAMD64LEAQ1}:  OpAMD64MOVQloadidx1,
	[2]Op{OpAMD64MOVQload, OpAMD64LEAQ8}:  OpAMD64MOVQloadidx8,
	[2]Op{OpAMD64MOVSSload, OpAMD64LEAQ1}: OpAMD64MOVSSloadidx1,
	[2]Op{OpAMD64MOVSSload, OpAMD64LEAQ4}: OpAMD64MOVSSloadidx4,
	[2]Op{OpAMD64MOVSDload, OpAMD64LEAQ1}: OpAMD64MOVSDloadidx1,
	[2]Op{OpAMD64MOVSDload, OpAMD64LEAQ8}: OpAMD64MOVSDloadidx8,

	[2]Op{OpAMD64MOVBstore, OpAMD64LEAQ1}:  OpAMD64MOVBstoreidx1,
	[2]Op{OpAMD64MOVWstore, OpAMD64LEAQ1}:  OpAMD64MOVWstoreidx1,
	[2]Op{OpAMD64MOVWstore, OpAMD64LEAQ2}:  OpAMD64MOVWstoreidx2,
	[2]Op{OpAMD64MOVLstore, OpAMD64LEAQ1}:  OpAMD64MOVLstoreidx1,
	[2]Op{OpAMD64MOVLstore, OpAMD64LEAQ4}:  OpAMD64MOVLstoreidx4,
	[2]Op{OpAMD64MOVLstore, OpAMD64LEAQ8}:  OpAMD64MOVLstoreidx8,
	[2]Op{OpAMD64MOVQstore, OpAMD64LEAQ1}:  OpAMD64MOVQstoreidx1,
	[2]Op{OpAMD64MOVQstore, OpAMD64LEAQ8}:  OpAMD64MOVQstoreidx8,
	[2]Op{OpAMD64MOVSSstore, OpAMD64LEAQ1}: OpAMD64MOVSSstoreidx1,
	[2]Op{OpAMD64MOVSSstore, OpAMD64LEAQ4}: OpAMD64MOVSSstoreidx4,
	[2]Op{OpAMD64MOVSDstore, OpAMD64LEAQ1}: OpAMD64MOVSDstoreidx1,
	[2]Op{OpAMD64MOVSDstore, OpAMD64LEAQ8}: OpAMD64MOVSDstoreidx8,

	[2]Op{OpAMD64MOVBstoreconst, OpAMD64LEAQ1}: OpAMD64MOVBstoreconstidx1,
	[2]Op{OpAMD64MOVWstoreconst, OpAMD64LEAQ1}: OpAMD64MOVWstoreconstidx1,
	[2]Op{OpAMD64MOVWstoreconst, OpAMD64LEAQ2}: OpAMD64MOVWstoreconstidx2,
	[2]Op{OpAMD64MOVLstoreconst, OpAMD64LEAQ1}: OpAMD64MOVLstoreconstidx1,
	[2]Op{OpAMD64MOVLstoreconst, OpAMD64LEAQ4}: OpAMD64MOVLstoreconstidx4,
	[2]Op{OpAMD64MOVQstoreconst, OpAMD64LEAQ1}: OpAMD64MOVQstoreconstidx1,
	[2]Op{OpAMD64MOVQstoreconst, OpAMD64LEAQ8}: OpAMD64MOVQstoreconstidx8,

	// 386
	[2]Op{Op386MOVBload, Op386ADDL}:  Op386MOVBloadidx1,
	[2]Op{Op386MOVWload, Op386ADDL}:  Op386MOVWloadidx1,
	[2]Op{Op386MOVLload, Op386ADDL}:  Op386MOVLloadidx1,
	[2]Op{Op386MOVSSload, Op386ADDL}: Op386MOVSSloadidx1,
	[2]Op{Op386MOVSDload, Op386ADDL}: Op386MOVSDloadidx1,

	[2]Op{Op386MOVBstore, Op386ADDL}:  Op386MOVBstoreidx1,
	[2]Op{Op386MOVWstore, Op386ADDL}:  Op386MOVWstoreidx1,
	[2]Op{Op386MOVLstore, Op386ADDL}:  Op386MOVLstoreidx1,
	[2]Op{Op386MOVSSstore, Op386ADDL}: Op386MOVSSstoreidx1,
	[2]Op{Op386MOVSDstore, Op386ADDL}: Op386MOVSDstoreidx1,

	[2]Op{Op386MOVBstoreconst, Op386ADDL}: Op386MOVBstoreconstidx1,
	[2]Op{Op386MOVWstoreconst, Op386ADDL}: Op386MOVWstoreconstidx1,
	[2]Op{Op386MOVLstoreconst, Op386ADDL}: Op386MOVLstoreconstidx1,

	[2]Op{Op386MOVBload, Op386LEAL1}:  Op386MOVBloadidx1,
	[2]Op{Op386MOVWload, Op386LEAL1}:  Op386MOVWloadidx1,
	[2]Op{Op386MOVWload, Op386LEAL2}:  Op386MOVWloadidx2,
	[2]Op{Op386MOVLload, Op386LEAL1}:  Op386MOVLloadidx1,
	[2]Op{Op386MOVLload, Op386LEAL4}:  Op386MOVLloadidx4,
	[2]Op{Op386MOVSSload, Op386LEAL1}: Op386MOVSSloadidx1,
	[2]Op{Op386MOVSSload, Op386LEAL4}: Op386MOVSSloadidx4,
	[2]Op{Op386MOVSDload, Op386LEAL1}: Op386MOVSDloadidx1,
	[2]Op{Op386MOVSDload, Op386LEAL8}: Op386MOVSDloadidx8,

	[2]Op{Op386MOVBstore, Op386LEAL1}:  Op386MOVBstoreidx1,
	[2]Op{Op386MOVWstore, Op386LEAL1}:  Op386MOVWstoreidx1,
	[2]Op{Op386MOVWstore, Op386LEAL2}:  Op386MOVWstoreidx2,
	[2]Op{Op386MOVLstore, Op386LEAL1}:  Op386MOVLstoreidx1,
	[2]Op{Op386MOVLstore, Op386LEAL4}:  Op386MOVLstoreidx4,
	[2]Op{Op386MOVSSstore, Op386LEAL1}: Op386MOVSSstoreidx1,
	[2]Op{Op386MOVSSstore, Op386LEAL4}: Op386MOVSSstoreidx4,
	[2]Op{Op386MOVSDstore, Op386LEAL1}: Op386MOVSDstoreidx1,
	[2]Op{Op386MOVSDstore, Op386LEAL8}: Op386MOVSDstoreidx8,

	[2]Op{Op386MOVBstoreconst, Op386LEAL1}: Op386MOVBstoreconstidx1,
	[2]Op{Op386MOVWstoreconst, Op386LEAL1}: Op386MOVWstoreconstidx1,
	[2]Op{Op386MOVWstoreconst, Op386LEAL2}: Op386MOVWstoreconstidx2,
	[2]Op{Op386MOVLstoreconst, Op386LEAL1}: Op386MOVLstoreconstidx1,
	[2]Op{Op386MOVLstoreconst, Op386LEAL4}: Op386MOVLstoreconstidx4,

	[2]Op{Op386ADDLload, Op386LEAL4}: Op386ADDLloadidx4,
	[2]Op{Op386SUBLload, Op386LEAL4}: Op386SUBLloadidx4,
	[2]Op{Op386MULLload, Op386LEAL4}: Op386MULLloadidx4,
	[2]Op{Op386ANDLload, Op386LEAL4}: Op386ANDLloadidx4,
	[2]Op{Op386ORLload, Op386LEAL4}:  Op386ORLloadidx4,
	[2]Op{Op386XORLload, Op386LEAL4}: Op386XORLloadidx4,

	[2]Op{Op386ADDLmodify, Op386LEAL4}: Op386ADDLmodifyidx4,
	[2]Op{Op386SUBLmodify, Op386LEAL4}: Op386SUBLmodifyidx4,
	[2]Op{Op386ANDLmodify, Op386LEAL4}: Op386ANDLmodifyidx4,
	[2]Op{Op386ORLmodify, Op386LEAL4}:  Op386ORLmodifyidx4,
	[2]Op{Op386XORLmodify, Op386LEAL4}: Op386XORLmodifyidx4,

	[2]Op{Op386ADDLconstmodify, Op386LEAL4}: Op386ADDLconstmodifyidx4,
	[2]Op{Op386ANDLconstmodify, Op386LEAL4}: Op386ANDLconstmodifyidx4,
	[2]Op{Op386ORLconstmodify, Op386LEAL4}:  Op386ORLconstmodifyidx4,
	[2]Op{Op386XORLconstmodify, Op386LEAL4}: Op386XORLconstmodifyidx4,
}
