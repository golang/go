// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// addressingModes combines address calculations into memory operations
// that can perform complicated addressing modes.
func addressingModes(f *Func) {
	isInImmediateRange := is32Bit
	switch f.Config.arch {
	default:
		// Most architectures can't do this.
		return
	case "amd64", "386":
	case "s390x":
		isInImmediateRange = is20Bit
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
				if !isInImmediateRange(v.AuxInt + p.AuxInt) {
					continue
				}
				v.AuxInt += p.AuxInt
			case [2]auxType{auxSymOff, auxSymOff}:
				if v.Aux != nil && p.Aux != nil {
					continue
				}
				if !isInImmediateRange(v.AuxInt + p.AuxInt) {
					continue
				}
				if p.Aux != nil {
					v.Aux = p.Aux
				}
				v.AuxInt += p.AuxInt
			case [2]auxType{auxSymValAndOff, auxInt32}:
				vo := ValAndOff(v.AuxInt)
				if !vo.canAdd64(p.AuxInt) {
					continue
				}
				v.AuxInt = int64(vo.addOffset64(p.AuxInt))
			case [2]auxType{auxSymValAndOff, auxSymOff}:
				vo := ValAndOff(v.AuxInt)
				if v.Aux != nil && p.Aux != nil {
					continue
				}
				if !vo.canAdd64(p.AuxInt) {
					continue
				}
				if p.Aux != nil {
					v.Aux = p.Aux
				}
				v.AuxInt = int64(vo.addOffset64(p.AuxInt))
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
			if needSplit[c] {
				// It turns out that some of the combined instructions have faster two-instruction equivalents,
				// but not the two instructions that led to them being combined here.  For example
				// (CMPBconstload c (ADDQ x y)) -> (CMPBconstloadidx1 c x y) -> (CMPB c (MOVBloadidx1 x y))
				// The final pair of instructions turns out to be notably faster, at least in some benchmarks.
				f.Config.splitLoad(v)
			}
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

// needSplit contains instructions that should be postprocessed by splitLoad
// into a more-efficient two-instruction form.
var needSplit = map[Op]bool{
	OpAMD64CMPBloadidx1: true,
	OpAMD64CMPWloadidx1: true,
	OpAMD64CMPLloadidx1: true,
	OpAMD64CMPQloadidx1: true,
	OpAMD64CMPWloadidx2: true,
	OpAMD64CMPLloadidx4: true,
	OpAMD64CMPQloadidx8: true,

	OpAMD64CMPBconstloadidx1: true,
	OpAMD64CMPWconstloadidx1: true,
	OpAMD64CMPLconstloadidx1: true,
	OpAMD64CMPQconstloadidx1: true,
	OpAMD64CMPWconstloadidx2: true,
	OpAMD64CMPLconstloadidx4: true,
	OpAMD64CMPQconstloadidx8: true,
}

// For each entry k, v in this map, if we have a value x with:
//
//	x.Op == k[0]
//	x.Args[0].Op == k[1]
//
// then we can set x.Op to v and set x.Args like this:
//
//	x.Args[0].Args + x.Args[1:]
//
// Additionally, the Aux/AuxInt from x.Args[0] is merged into x.
var combine = map[[2]Op]Op{
	// amd64
	{OpAMD64MOVBload, OpAMD64ADDQ}:  OpAMD64MOVBloadidx1,
	{OpAMD64MOVWload, OpAMD64ADDQ}:  OpAMD64MOVWloadidx1,
	{OpAMD64MOVLload, OpAMD64ADDQ}:  OpAMD64MOVLloadidx1,
	{OpAMD64MOVQload, OpAMD64ADDQ}:  OpAMD64MOVQloadidx1,
	{OpAMD64MOVSSload, OpAMD64ADDQ}: OpAMD64MOVSSloadidx1,
	{OpAMD64MOVSDload, OpAMD64ADDQ}: OpAMD64MOVSDloadidx1,

	{OpAMD64MOVBstore, OpAMD64ADDQ}:  OpAMD64MOVBstoreidx1,
	{OpAMD64MOVWstore, OpAMD64ADDQ}:  OpAMD64MOVWstoreidx1,
	{OpAMD64MOVLstore, OpAMD64ADDQ}:  OpAMD64MOVLstoreidx1,
	{OpAMD64MOVQstore, OpAMD64ADDQ}:  OpAMD64MOVQstoreidx1,
	{OpAMD64MOVSSstore, OpAMD64ADDQ}: OpAMD64MOVSSstoreidx1,
	{OpAMD64MOVSDstore, OpAMD64ADDQ}: OpAMD64MOVSDstoreidx1,

	{OpAMD64MOVBstoreconst, OpAMD64ADDQ}: OpAMD64MOVBstoreconstidx1,
	{OpAMD64MOVWstoreconst, OpAMD64ADDQ}: OpAMD64MOVWstoreconstidx1,
	{OpAMD64MOVLstoreconst, OpAMD64ADDQ}: OpAMD64MOVLstoreconstidx1,
	{OpAMD64MOVQstoreconst, OpAMD64ADDQ}: OpAMD64MOVQstoreconstidx1,

	{OpAMD64MOVBload, OpAMD64LEAQ1}:  OpAMD64MOVBloadidx1,
	{OpAMD64MOVWload, OpAMD64LEAQ1}:  OpAMD64MOVWloadidx1,
	{OpAMD64MOVWload, OpAMD64LEAQ2}:  OpAMD64MOVWloadidx2,
	{OpAMD64MOVLload, OpAMD64LEAQ1}:  OpAMD64MOVLloadidx1,
	{OpAMD64MOVLload, OpAMD64LEAQ4}:  OpAMD64MOVLloadidx4,
	{OpAMD64MOVLload, OpAMD64LEAQ8}:  OpAMD64MOVLloadidx8,
	{OpAMD64MOVQload, OpAMD64LEAQ1}:  OpAMD64MOVQloadidx1,
	{OpAMD64MOVQload, OpAMD64LEAQ8}:  OpAMD64MOVQloadidx8,
	{OpAMD64MOVSSload, OpAMD64LEAQ1}: OpAMD64MOVSSloadidx1,
	{OpAMD64MOVSSload, OpAMD64LEAQ4}: OpAMD64MOVSSloadidx4,
	{OpAMD64MOVSDload, OpAMD64LEAQ1}: OpAMD64MOVSDloadidx1,
	{OpAMD64MOVSDload, OpAMD64LEAQ8}: OpAMD64MOVSDloadidx8,

	{OpAMD64MOVBstore, OpAMD64LEAQ1}:  OpAMD64MOVBstoreidx1,
	{OpAMD64MOVWstore, OpAMD64LEAQ1}:  OpAMD64MOVWstoreidx1,
	{OpAMD64MOVWstore, OpAMD64LEAQ2}:  OpAMD64MOVWstoreidx2,
	{OpAMD64MOVLstore, OpAMD64LEAQ1}:  OpAMD64MOVLstoreidx1,
	{OpAMD64MOVLstore, OpAMD64LEAQ4}:  OpAMD64MOVLstoreidx4,
	{OpAMD64MOVLstore, OpAMD64LEAQ8}:  OpAMD64MOVLstoreidx8,
	{OpAMD64MOVQstore, OpAMD64LEAQ1}:  OpAMD64MOVQstoreidx1,
	{OpAMD64MOVQstore, OpAMD64LEAQ8}:  OpAMD64MOVQstoreidx8,
	{OpAMD64MOVSSstore, OpAMD64LEAQ1}: OpAMD64MOVSSstoreidx1,
	{OpAMD64MOVSSstore, OpAMD64LEAQ4}: OpAMD64MOVSSstoreidx4,
	{OpAMD64MOVSDstore, OpAMD64LEAQ1}: OpAMD64MOVSDstoreidx1,
	{OpAMD64MOVSDstore, OpAMD64LEAQ8}: OpAMD64MOVSDstoreidx8,

	{OpAMD64MOVBstoreconst, OpAMD64LEAQ1}: OpAMD64MOVBstoreconstidx1,
	{OpAMD64MOVWstoreconst, OpAMD64LEAQ1}: OpAMD64MOVWstoreconstidx1,
	{OpAMD64MOVWstoreconst, OpAMD64LEAQ2}: OpAMD64MOVWstoreconstidx2,
	{OpAMD64MOVLstoreconst, OpAMD64LEAQ1}: OpAMD64MOVLstoreconstidx1,
	{OpAMD64MOVLstoreconst, OpAMD64LEAQ4}: OpAMD64MOVLstoreconstidx4,
	{OpAMD64MOVQstoreconst, OpAMD64LEAQ1}: OpAMD64MOVQstoreconstidx1,
	{OpAMD64MOVQstoreconst, OpAMD64LEAQ8}: OpAMD64MOVQstoreconstidx8,

	{OpAMD64SETEQstore, OpAMD64LEAQ1}: OpAMD64SETEQstoreidx1,
	{OpAMD64SETNEstore, OpAMD64LEAQ1}: OpAMD64SETNEstoreidx1,
	{OpAMD64SETLstore, OpAMD64LEAQ1}:  OpAMD64SETLstoreidx1,
	{OpAMD64SETLEstore, OpAMD64LEAQ1}: OpAMD64SETLEstoreidx1,
	{OpAMD64SETGstore, OpAMD64LEAQ1}:  OpAMD64SETGstoreidx1,
	{OpAMD64SETGEstore, OpAMD64LEAQ1}: OpAMD64SETGEstoreidx1,
	{OpAMD64SETBstore, OpAMD64LEAQ1}:  OpAMD64SETBstoreidx1,
	{OpAMD64SETBEstore, OpAMD64LEAQ1}: OpAMD64SETBEstoreidx1,
	{OpAMD64SETAstore, OpAMD64LEAQ1}:  OpAMD64SETAstoreidx1,
	{OpAMD64SETAEstore, OpAMD64LEAQ1}: OpAMD64SETAEstoreidx1,

	// These instructions are re-split differently for performance, see needSplit above.
	// TODO if 386 versions are created, also update needSplit and _gen/386splitload.rules
	{OpAMD64CMPBload, OpAMD64ADDQ}: OpAMD64CMPBloadidx1,
	{OpAMD64CMPWload, OpAMD64ADDQ}: OpAMD64CMPWloadidx1,
	{OpAMD64CMPLload, OpAMD64ADDQ}: OpAMD64CMPLloadidx1,
	{OpAMD64CMPQload, OpAMD64ADDQ}: OpAMD64CMPQloadidx1,

	{OpAMD64CMPBload, OpAMD64LEAQ1}: OpAMD64CMPBloadidx1,
	{OpAMD64CMPWload, OpAMD64LEAQ1}: OpAMD64CMPWloadidx1,
	{OpAMD64CMPWload, OpAMD64LEAQ2}: OpAMD64CMPWloadidx2,
	{OpAMD64CMPLload, OpAMD64LEAQ1}: OpAMD64CMPLloadidx1,
	{OpAMD64CMPLload, OpAMD64LEAQ4}: OpAMD64CMPLloadidx4,
	{OpAMD64CMPQload, OpAMD64LEAQ1}: OpAMD64CMPQloadidx1,
	{OpAMD64CMPQload, OpAMD64LEAQ8}: OpAMD64CMPQloadidx8,

	{OpAMD64CMPBconstload, OpAMD64ADDQ}: OpAMD64CMPBconstloadidx1,
	{OpAMD64CMPWconstload, OpAMD64ADDQ}: OpAMD64CMPWconstloadidx1,
	{OpAMD64CMPLconstload, OpAMD64ADDQ}: OpAMD64CMPLconstloadidx1,
	{OpAMD64CMPQconstload, OpAMD64ADDQ}: OpAMD64CMPQconstloadidx1,

	{OpAMD64CMPBconstload, OpAMD64LEAQ1}: OpAMD64CMPBconstloadidx1,
	{OpAMD64CMPWconstload, OpAMD64LEAQ1}: OpAMD64CMPWconstloadidx1,
	{OpAMD64CMPWconstload, OpAMD64LEAQ2}: OpAMD64CMPWconstloadidx2,
	{OpAMD64CMPLconstload, OpAMD64LEAQ1}: OpAMD64CMPLconstloadidx1,
	{OpAMD64CMPLconstload, OpAMD64LEAQ4}: OpAMD64CMPLconstloadidx4,
	{OpAMD64CMPQconstload, OpAMD64LEAQ1}: OpAMD64CMPQconstloadidx1,
	{OpAMD64CMPQconstload, OpAMD64LEAQ8}: OpAMD64CMPQconstloadidx8,

	{OpAMD64ADDLload, OpAMD64ADDQ}: OpAMD64ADDLloadidx1,
	{OpAMD64ADDQload, OpAMD64ADDQ}: OpAMD64ADDQloadidx1,
	{OpAMD64SUBLload, OpAMD64ADDQ}: OpAMD64SUBLloadidx1,
	{OpAMD64SUBQload, OpAMD64ADDQ}: OpAMD64SUBQloadidx1,
	{OpAMD64ANDLload, OpAMD64ADDQ}: OpAMD64ANDLloadidx1,
	{OpAMD64ANDQload, OpAMD64ADDQ}: OpAMD64ANDQloadidx1,
	{OpAMD64ORLload, OpAMD64ADDQ}:  OpAMD64ORLloadidx1,
	{OpAMD64ORQload, OpAMD64ADDQ}:  OpAMD64ORQloadidx1,
	{OpAMD64XORLload, OpAMD64ADDQ}: OpAMD64XORLloadidx1,
	{OpAMD64XORQload, OpAMD64ADDQ}: OpAMD64XORQloadidx1,

	{OpAMD64ADDLload, OpAMD64LEAQ1}: OpAMD64ADDLloadidx1,
	{OpAMD64ADDLload, OpAMD64LEAQ4}: OpAMD64ADDLloadidx4,
	{OpAMD64ADDLload, OpAMD64LEAQ8}: OpAMD64ADDLloadidx8,
	{OpAMD64ADDQload, OpAMD64LEAQ1}: OpAMD64ADDQloadidx1,
	{OpAMD64ADDQload, OpAMD64LEAQ8}: OpAMD64ADDQloadidx8,
	{OpAMD64SUBLload, OpAMD64LEAQ1}: OpAMD64SUBLloadidx1,
	{OpAMD64SUBLload, OpAMD64LEAQ4}: OpAMD64SUBLloadidx4,
	{OpAMD64SUBLload, OpAMD64LEAQ8}: OpAMD64SUBLloadidx8,
	{OpAMD64SUBQload, OpAMD64LEAQ1}: OpAMD64SUBQloadidx1,
	{OpAMD64SUBQload, OpAMD64LEAQ8}: OpAMD64SUBQloadidx8,
	{OpAMD64ANDLload, OpAMD64LEAQ1}: OpAMD64ANDLloadidx1,
	{OpAMD64ANDLload, OpAMD64LEAQ4}: OpAMD64ANDLloadidx4,
	{OpAMD64ANDLload, OpAMD64LEAQ8}: OpAMD64ANDLloadidx8,
	{OpAMD64ANDQload, OpAMD64LEAQ1}: OpAMD64ANDQloadidx1,
	{OpAMD64ANDQload, OpAMD64LEAQ8}: OpAMD64ANDQloadidx8,
	{OpAMD64ORLload, OpAMD64LEAQ1}:  OpAMD64ORLloadidx1,
	{OpAMD64ORLload, OpAMD64LEAQ4}:  OpAMD64ORLloadidx4,
	{OpAMD64ORLload, OpAMD64LEAQ8}:  OpAMD64ORLloadidx8,
	{OpAMD64ORQload, OpAMD64LEAQ1}:  OpAMD64ORQloadidx1,
	{OpAMD64ORQload, OpAMD64LEAQ8}:  OpAMD64ORQloadidx8,
	{OpAMD64XORLload, OpAMD64LEAQ1}: OpAMD64XORLloadidx1,
	{OpAMD64XORLload, OpAMD64LEAQ4}: OpAMD64XORLloadidx4,
	{OpAMD64XORLload, OpAMD64LEAQ8}: OpAMD64XORLloadidx8,
	{OpAMD64XORQload, OpAMD64LEAQ1}: OpAMD64XORQloadidx1,
	{OpAMD64XORQload, OpAMD64LEAQ8}: OpAMD64XORQloadidx8,

	{OpAMD64ADDLmodify, OpAMD64ADDQ}: OpAMD64ADDLmodifyidx1,
	{OpAMD64ADDQmodify, OpAMD64ADDQ}: OpAMD64ADDQmodifyidx1,
	{OpAMD64SUBLmodify, OpAMD64ADDQ}: OpAMD64SUBLmodifyidx1,
	{OpAMD64SUBQmodify, OpAMD64ADDQ}: OpAMD64SUBQmodifyidx1,
	{OpAMD64ANDLmodify, OpAMD64ADDQ}: OpAMD64ANDLmodifyidx1,
	{OpAMD64ANDQmodify, OpAMD64ADDQ}: OpAMD64ANDQmodifyidx1,
	{OpAMD64ORLmodify, OpAMD64ADDQ}:  OpAMD64ORLmodifyidx1,
	{OpAMD64ORQmodify, OpAMD64ADDQ}:  OpAMD64ORQmodifyidx1,
	{OpAMD64XORLmodify, OpAMD64ADDQ}: OpAMD64XORLmodifyidx1,
	{OpAMD64XORQmodify, OpAMD64ADDQ}: OpAMD64XORQmodifyidx1,

	{OpAMD64ADDLmodify, OpAMD64LEAQ1}: OpAMD64ADDLmodifyidx1,
	{OpAMD64ADDLmodify, OpAMD64LEAQ4}: OpAMD64ADDLmodifyidx4,
	{OpAMD64ADDLmodify, OpAMD64LEAQ8}: OpAMD64ADDLmodifyidx8,
	{OpAMD64ADDQmodify, OpAMD64LEAQ1}: OpAMD64ADDQmodifyidx1,
	{OpAMD64ADDQmodify, OpAMD64LEAQ8}: OpAMD64ADDQmodifyidx8,
	{OpAMD64SUBLmodify, OpAMD64LEAQ1}: OpAMD64SUBLmodifyidx1,
	{OpAMD64SUBLmodify, OpAMD64LEAQ4}: OpAMD64SUBLmodifyidx4,
	{OpAMD64SUBLmodify, OpAMD64LEAQ8}: OpAMD64SUBLmodifyidx8,
	{OpAMD64SUBQmodify, OpAMD64LEAQ1}: OpAMD64SUBQmodifyidx1,
	{OpAMD64SUBQmodify, OpAMD64LEAQ8}: OpAMD64SUBQmodifyidx8,
	{OpAMD64ANDLmodify, OpAMD64LEAQ1}: OpAMD64ANDLmodifyidx1,
	{OpAMD64ANDLmodify, OpAMD64LEAQ4}: OpAMD64ANDLmodifyidx4,
	{OpAMD64ANDLmodify, OpAMD64LEAQ8}: OpAMD64ANDLmodifyidx8,
	{OpAMD64ANDQmodify, OpAMD64LEAQ1}: OpAMD64ANDQmodifyidx1,
	{OpAMD64ANDQmodify, OpAMD64LEAQ8}: OpAMD64ANDQmodifyidx8,
	{OpAMD64ORLmodify, OpAMD64LEAQ1}:  OpAMD64ORLmodifyidx1,
	{OpAMD64ORLmodify, OpAMD64LEAQ4}:  OpAMD64ORLmodifyidx4,
	{OpAMD64ORLmodify, OpAMD64LEAQ8}:  OpAMD64ORLmodifyidx8,
	{OpAMD64ORQmodify, OpAMD64LEAQ1}:  OpAMD64ORQmodifyidx1,
	{OpAMD64ORQmodify, OpAMD64LEAQ8}:  OpAMD64ORQmodifyidx8,
	{OpAMD64XORLmodify, OpAMD64LEAQ1}: OpAMD64XORLmodifyidx1,
	{OpAMD64XORLmodify, OpAMD64LEAQ4}: OpAMD64XORLmodifyidx4,
	{OpAMD64XORLmodify, OpAMD64LEAQ8}: OpAMD64XORLmodifyidx8,
	{OpAMD64XORQmodify, OpAMD64LEAQ1}: OpAMD64XORQmodifyidx1,
	{OpAMD64XORQmodify, OpAMD64LEAQ8}: OpAMD64XORQmodifyidx8,

	{OpAMD64ADDLconstmodify, OpAMD64ADDQ}: OpAMD64ADDLconstmodifyidx1,
	{OpAMD64ADDQconstmodify, OpAMD64ADDQ}: OpAMD64ADDQconstmodifyidx1,
	{OpAMD64ANDLconstmodify, OpAMD64ADDQ}: OpAMD64ANDLconstmodifyidx1,
	{OpAMD64ANDQconstmodify, OpAMD64ADDQ}: OpAMD64ANDQconstmodifyidx1,
	{OpAMD64ORLconstmodify, OpAMD64ADDQ}:  OpAMD64ORLconstmodifyidx1,
	{OpAMD64ORQconstmodify, OpAMD64ADDQ}:  OpAMD64ORQconstmodifyidx1,
	{OpAMD64XORLconstmodify, OpAMD64ADDQ}: OpAMD64XORLconstmodifyidx1,
	{OpAMD64XORQconstmodify, OpAMD64ADDQ}: OpAMD64XORQconstmodifyidx1,

	{OpAMD64ADDLconstmodify, OpAMD64LEAQ1}: OpAMD64ADDLconstmodifyidx1,
	{OpAMD64ADDLconstmodify, OpAMD64LEAQ4}: OpAMD64ADDLconstmodifyidx4,
	{OpAMD64ADDLconstmodify, OpAMD64LEAQ8}: OpAMD64ADDLconstmodifyidx8,
	{OpAMD64ADDQconstmodify, OpAMD64LEAQ1}: OpAMD64ADDQconstmodifyidx1,
	{OpAMD64ADDQconstmodify, OpAMD64LEAQ8}: OpAMD64ADDQconstmodifyidx8,
	{OpAMD64ANDLconstmodify, OpAMD64LEAQ1}: OpAMD64ANDLconstmodifyidx1,
	{OpAMD64ANDLconstmodify, OpAMD64LEAQ4}: OpAMD64ANDLconstmodifyidx4,
	{OpAMD64ANDLconstmodify, OpAMD64LEAQ8}: OpAMD64ANDLconstmodifyidx8,
	{OpAMD64ANDQconstmodify, OpAMD64LEAQ1}: OpAMD64ANDQconstmodifyidx1,
	{OpAMD64ANDQconstmodify, OpAMD64LEAQ8}: OpAMD64ANDQconstmodifyidx8,
	{OpAMD64ORLconstmodify, OpAMD64LEAQ1}:  OpAMD64ORLconstmodifyidx1,
	{OpAMD64ORLconstmodify, OpAMD64LEAQ4}:  OpAMD64ORLconstmodifyidx4,
	{OpAMD64ORLconstmodify, OpAMD64LEAQ8}:  OpAMD64ORLconstmodifyidx8,
	{OpAMD64ORQconstmodify, OpAMD64LEAQ1}:  OpAMD64ORQconstmodifyidx1,
	{OpAMD64ORQconstmodify, OpAMD64LEAQ8}:  OpAMD64ORQconstmodifyidx8,
	{OpAMD64XORLconstmodify, OpAMD64LEAQ1}: OpAMD64XORLconstmodifyidx1,
	{OpAMD64XORLconstmodify, OpAMD64LEAQ4}: OpAMD64XORLconstmodifyidx4,
	{OpAMD64XORLconstmodify, OpAMD64LEAQ8}: OpAMD64XORLconstmodifyidx8,
	{OpAMD64XORQconstmodify, OpAMD64LEAQ1}: OpAMD64XORQconstmodifyidx1,
	{OpAMD64XORQconstmodify, OpAMD64LEAQ8}: OpAMD64XORQconstmodifyidx8,

	{OpAMD64ADDSSload, OpAMD64LEAQ1}: OpAMD64ADDSSloadidx1,
	{OpAMD64ADDSSload, OpAMD64LEAQ4}: OpAMD64ADDSSloadidx4,
	{OpAMD64ADDSDload, OpAMD64LEAQ1}: OpAMD64ADDSDloadidx1,
	{OpAMD64ADDSDload, OpAMD64LEAQ8}: OpAMD64ADDSDloadidx8,
	{OpAMD64SUBSSload, OpAMD64LEAQ1}: OpAMD64SUBSSloadidx1,
	{OpAMD64SUBSSload, OpAMD64LEAQ4}: OpAMD64SUBSSloadidx4,
	{OpAMD64SUBSDload, OpAMD64LEAQ1}: OpAMD64SUBSDloadidx1,
	{OpAMD64SUBSDload, OpAMD64LEAQ8}: OpAMD64SUBSDloadidx8,
	{OpAMD64MULSSload, OpAMD64LEAQ1}: OpAMD64MULSSloadidx1,
	{OpAMD64MULSSload, OpAMD64LEAQ4}: OpAMD64MULSSloadidx4,
	{OpAMD64MULSDload, OpAMD64LEAQ1}: OpAMD64MULSDloadidx1,
	{OpAMD64MULSDload, OpAMD64LEAQ8}: OpAMD64MULSDloadidx8,
	{OpAMD64DIVSSload, OpAMD64LEAQ1}: OpAMD64DIVSSloadidx1,
	{OpAMD64DIVSSload, OpAMD64LEAQ4}: OpAMD64DIVSSloadidx4,
	{OpAMD64DIVSDload, OpAMD64LEAQ1}: OpAMD64DIVSDloadidx1,
	{OpAMD64DIVSDload, OpAMD64LEAQ8}: OpAMD64DIVSDloadidx8,

	{OpAMD64SARXLload, OpAMD64ADDQ}: OpAMD64SARXLloadidx1,
	{OpAMD64SARXQload, OpAMD64ADDQ}: OpAMD64SARXQloadidx1,
	{OpAMD64SHLXLload, OpAMD64ADDQ}: OpAMD64SHLXLloadidx1,
	{OpAMD64SHLXQload, OpAMD64ADDQ}: OpAMD64SHLXQloadidx1,
	{OpAMD64SHRXLload, OpAMD64ADDQ}: OpAMD64SHRXLloadidx1,
	{OpAMD64SHRXQload, OpAMD64ADDQ}: OpAMD64SHRXQloadidx1,

	{OpAMD64SARXLload, OpAMD64LEAQ1}: OpAMD64SARXLloadidx1,
	{OpAMD64SARXLload, OpAMD64LEAQ4}: OpAMD64SARXLloadidx4,
	{OpAMD64SARXLload, OpAMD64LEAQ8}: OpAMD64SARXLloadidx8,
	{OpAMD64SARXQload, OpAMD64LEAQ1}: OpAMD64SARXQloadidx1,
	{OpAMD64SARXQload, OpAMD64LEAQ8}: OpAMD64SARXQloadidx8,
	{OpAMD64SHLXLload, OpAMD64LEAQ1}: OpAMD64SHLXLloadidx1,
	{OpAMD64SHLXLload, OpAMD64LEAQ4}: OpAMD64SHLXLloadidx4,
	{OpAMD64SHLXLload, OpAMD64LEAQ8}: OpAMD64SHLXLloadidx8,
	{OpAMD64SHLXQload, OpAMD64LEAQ1}: OpAMD64SHLXQloadidx1,
	{OpAMD64SHLXQload, OpAMD64LEAQ8}: OpAMD64SHLXQloadidx8,
	{OpAMD64SHRXLload, OpAMD64LEAQ1}: OpAMD64SHRXLloadidx1,
	{OpAMD64SHRXLload, OpAMD64LEAQ4}: OpAMD64SHRXLloadidx4,
	{OpAMD64SHRXLload, OpAMD64LEAQ8}: OpAMD64SHRXLloadidx8,
	{OpAMD64SHRXQload, OpAMD64LEAQ1}: OpAMD64SHRXQloadidx1,
	{OpAMD64SHRXQload, OpAMD64LEAQ8}: OpAMD64SHRXQloadidx8,

	// amd64/v3
	{OpAMD64MOVBELload, OpAMD64ADDQ}:  OpAMD64MOVBELloadidx1,
	{OpAMD64MOVBEQload, OpAMD64ADDQ}:  OpAMD64MOVBEQloadidx1,
	{OpAMD64MOVBELload, OpAMD64LEAQ1}: OpAMD64MOVBELloadidx1,
	{OpAMD64MOVBELload, OpAMD64LEAQ4}: OpAMD64MOVBELloadidx4,
	{OpAMD64MOVBELload, OpAMD64LEAQ8}: OpAMD64MOVBELloadidx8,
	{OpAMD64MOVBEQload, OpAMD64LEAQ1}: OpAMD64MOVBEQloadidx1,
	{OpAMD64MOVBEQload, OpAMD64LEAQ8}: OpAMD64MOVBEQloadidx8,

	{OpAMD64MOVBEWstore, OpAMD64ADDQ}:  OpAMD64MOVBEWstoreidx1,
	{OpAMD64MOVBELstore, OpAMD64ADDQ}:  OpAMD64MOVBELstoreidx1,
	{OpAMD64MOVBEQstore, OpAMD64ADDQ}:  OpAMD64MOVBEQstoreidx1,
	{OpAMD64MOVBEWstore, OpAMD64LEAQ1}: OpAMD64MOVBEWstoreidx1,
	{OpAMD64MOVBEWstore, OpAMD64LEAQ2}: OpAMD64MOVBEWstoreidx2,
	{OpAMD64MOVBELstore, OpAMD64LEAQ1}: OpAMD64MOVBELstoreidx1,
	{OpAMD64MOVBELstore, OpAMD64LEAQ4}: OpAMD64MOVBELstoreidx4,
	{OpAMD64MOVBELstore, OpAMD64LEAQ8}: OpAMD64MOVBELstoreidx8,
	{OpAMD64MOVBEQstore, OpAMD64LEAQ1}: OpAMD64MOVBEQstoreidx1,
	{OpAMD64MOVBEQstore, OpAMD64LEAQ8}: OpAMD64MOVBEQstoreidx8,

	// 386
	{Op386MOVBload, Op386ADDL}:  Op386MOVBloadidx1,
	{Op386MOVWload, Op386ADDL}:  Op386MOVWloadidx1,
	{Op386MOVLload, Op386ADDL}:  Op386MOVLloadidx1,
	{Op386MOVSSload, Op386ADDL}: Op386MOVSSloadidx1,
	{Op386MOVSDload, Op386ADDL}: Op386MOVSDloadidx1,

	{Op386MOVBstore, Op386ADDL}:  Op386MOVBstoreidx1,
	{Op386MOVWstore, Op386ADDL}:  Op386MOVWstoreidx1,
	{Op386MOVLstore, Op386ADDL}:  Op386MOVLstoreidx1,
	{Op386MOVSSstore, Op386ADDL}: Op386MOVSSstoreidx1,
	{Op386MOVSDstore, Op386ADDL}: Op386MOVSDstoreidx1,

	{Op386MOVBstoreconst, Op386ADDL}: Op386MOVBstoreconstidx1,
	{Op386MOVWstoreconst, Op386ADDL}: Op386MOVWstoreconstidx1,
	{Op386MOVLstoreconst, Op386ADDL}: Op386MOVLstoreconstidx1,

	{Op386MOVBload, Op386LEAL1}:  Op386MOVBloadidx1,
	{Op386MOVWload, Op386LEAL1}:  Op386MOVWloadidx1,
	{Op386MOVWload, Op386LEAL2}:  Op386MOVWloadidx2,
	{Op386MOVLload, Op386LEAL1}:  Op386MOVLloadidx1,
	{Op386MOVLload, Op386LEAL4}:  Op386MOVLloadidx4,
	{Op386MOVSSload, Op386LEAL1}: Op386MOVSSloadidx1,
	{Op386MOVSSload, Op386LEAL4}: Op386MOVSSloadidx4,
	{Op386MOVSDload, Op386LEAL1}: Op386MOVSDloadidx1,
	{Op386MOVSDload, Op386LEAL8}: Op386MOVSDloadidx8,

	{Op386MOVBstore, Op386LEAL1}:  Op386MOVBstoreidx1,
	{Op386MOVWstore, Op386LEAL1}:  Op386MOVWstoreidx1,
	{Op386MOVWstore, Op386LEAL2}:  Op386MOVWstoreidx2,
	{Op386MOVLstore, Op386LEAL1}:  Op386MOVLstoreidx1,
	{Op386MOVLstore, Op386LEAL4}:  Op386MOVLstoreidx4,
	{Op386MOVSSstore, Op386LEAL1}: Op386MOVSSstoreidx1,
	{Op386MOVSSstore, Op386LEAL4}: Op386MOVSSstoreidx4,
	{Op386MOVSDstore, Op386LEAL1}: Op386MOVSDstoreidx1,
	{Op386MOVSDstore, Op386LEAL8}: Op386MOVSDstoreidx8,

	{Op386MOVBstoreconst, Op386LEAL1}: Op386MOVBstoreconstidx1,
	{Op386MOVWstoreconst, Op386LEAL1}: Op386MOVWstoreconstidx1,
	{Op386MOVWstoreconst, Op386LEAL2}: Op386MOVWstoreconstidx2,
	{Op386MOVLstoreconst, Op386LEAL1}: Op386MOVLstoreconstidx1,
	{Op386MOVLstoreconst, Op386LEAL4}: Op386MOVLstoreconstidx4,

	{Op386ADDLload, Op386LEAL4}: Op386ADDLloadidx4,
	{Op386SUBLload, Op386LEAL4}: Op386SUBLloadidx4,
	{Op386MULLload, Op386LEAL4}: Op386MULLloadidx4,
	{Op386ANDLload, Op386LEAL4}: Op386ANDLloadidx4,
	{Op386ORLload, Op386LEAL4}:  Op386ORLloadidx4,
	{Op386XORLload, Op386LEAL4}: Op386XORLloadidx4,

	{Op386ADDLmodify, Op386LEAL4}: Op386ADDLmodifyidx4,
	{Op386SUBLmodify, Op386LEAL4}: Op386SUBLmodifyidx4,
	{Op386ANDLmodify, Op386LEAL4}: Op386ANDLmodifyidx4,
	{Op386ORLmodify, Op386LEAL4}:  Op386ORLmodifyidx4,
	{Op386XORLmodify, Op386LEAL4}: Op386XORLmodifyidx4,

	{Op386ADDLconstmodify, Op386LEAL4}: Op386ADDLconstmodifyidx4,
	{Op386ANDLconstmodify, Op386LEAL4}: Op386ANDLconstmodifyidx4,
	{Op386ORLconstmodify, Op386LEAL4}:  Op386ORLconstmodifyidx4,
	{Op386XORLconstmodify, Op386LEAL4}: Op386XORLconstmodifyidx4,

	// s390x
	{OpS390XMOVDload, OpS390XADD}: OpS390XMOVDloadidx,
	{OpS390XMOVWload, OpS390XADD}: OpS390XMOVWloadidx,
	{OpS390XMOVHload, OpS390XADD}: OpS390XMOVHloadidx,
	{OpS390XMOVBload, OpS390XADD}: OpS390XMOVBloadidx,

	{OpS390XMOVWZload, OpS390XADD}: OpS390XMOVWZloadidx,
	{OpS390XMOVHZload, OpS390XADD}: OpS390XMOVHZloadidx,
	{OpS390XMOVBZload, OpS390XADD}: OpS390XMOVBZloadidx,

	{OpS390XMOVDBRload, OpS390XADD}: OpS390XMOVDBRloadidx,
	{OpS390XMOVWBRload, OpS390XADD}: OpS390XMOVWBRloadidx,
	{OpS390XMOVHBRload, OpS390XADD}: OpS390XMOVHBRloadidx,

	{OpS390XFMOVDload, OpS390XADD}: OpS390XFMOVDloadidx,
	{OpS390XFMOVSload, OpS390XADD}: OpS390XFMOVSloadidx,

	{OpS390XMOVDstore, OpS390XADD}: OpS390XMOVDstoreidx,
	{OpS390XMOVWstore, OpS390XADD}: OpS390XMOVWstoreidx,
	{OpS390XMOVHstore, OpS390XADD}: OpS390XMOVHstoreidx,
	{OpS390XMOVBstore, OpS390XADD}: OpS390XMOVBstoreidx,

	{OpS390XMOVDBRstore, OpS390XADD}: OpS390XMOVDBRstoreidx,
	{OpS390XMOVWBRstore, OpS390XADD}: OpS390XMOVWBRstoreidx,
	{OpS390XMOVHBRstore, OpS390XADD}: OpS390XMOVHBRstoreidx,

	{OpS390XFMOVDstore, OpS390XADD}: OpS390XFMOVDstoreidx,
	{OpS390XFMOVSstore, OpS390XADD}: OpS390XFMOVSstoreidx,

	{OpS390XMOVDload, OpS390XMOVDaddridx}: OpS390XMOVDloadidx,
	{OpS390XMOVWload, OpS390XMOVDaddridx}: OpS390XMOVWloadidx,
	{OpS390XMOVHload, OpS390XMOVDaddridx}: OpS390XMOVHloadidx,
	{OpS390XMOVBload, OpS390XMOVDaddridx}: OpS390XMOVBloadidx,

	{OpS390XMOVWZload, OpS390XMOVDaddridx}: OpS390XMOVWZloadidx,
	{OpS390XMOVHZload, OpS390XMOVDaddridx}: OpS390XMOVHZloadidx,
	{OpS390XMOVBZload, OpS390XMOVDaddridx}: OpS390XMOVBZloadidx,

	{OpS390XMOVDBRload, OpS390XMOVDaddridx}: OpS390XMOVDBRloadidx,
	{OpS390XMOVWBRload, OpS390XMOVDaddridx}: OpS390XMOVWBRloadidx,
	{OpS390XMOVHBRload, OpS390XMOVDaddridx}: OpS390XMOVHBRloadidx,

	{OpS390XFMOVDload, OpS390XMOVDaddridx}: OpS390XFMOVDloadidx,
	{OpS390XFMOVSload, OpS390XMOVDaddridx}: OpS390XFMOVSloadidx,

	{OpS390XMOVDstore, OpS390XMOVDaddridx}: OpS390XMOVDstoreidx,
	{OpS390XMOVWstore, OpS390XMOVDaddridx}: OpS390XMOVWstoreidx,
	{OpS390XMOVHstore, OpS390XMOVDaddridx}: OpS390XMOVHstoreidx,
	{OpS390XMOVBstore, OpS390XMOVDaddridx}: OpS390XMOVBstoreidx,

	{OpS390XMOVDBRstore, OpS390XMOVDaddridx}: OpS390XMOVDBRstoreidx,
	{OpS390XMOVWBRstore, OpS390XMOVDaddridx}: OpS390XMOVWBRstoreidx,
	{OpS390XMOVHBRstore, OpS390XMOVDaddridx}: OpS390XMOVHBRstoreidx,

	{OpS390XFMOVDstore, OpS390XMOVDaddridx}: OpS390XFMOVDstoreidx,
	{OpS390XFMOVSstore, OpS390XMOVDaddridx}: OpS390XFMOVSstoreidx,
}
