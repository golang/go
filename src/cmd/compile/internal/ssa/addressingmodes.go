// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/ssa/ssacore"
	"cmd/compile/internal/ssa/ssaop"
)

// addressingModes combines address calculations into memory operations
// that can perform complicated addressing modes.
func addressingModes(f *ssacore.Func) {
	isInImmediateRange := Is32Bit
	switch f.Config.Arch {
	default:
		// Most architectures can't do this.
		return
	case "amd64", "386":
	case "s390x":
		isInImmediateRange = Is20Bit
	}

	var tmp []*ssacore.Value
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
			if ssaop.OpcodeTable[v.Op].ResultInArg0 {
				ptrIndex = 1
			}
			p := v.Args[ptrIndex]
			c, ok := combine[[2]ssaop.Op{v.Op, p.Op}]
			if !ok {
				continue
			}
			// See if we can combine the Aux/AuxInt values.
			switch [2]ssaop.AuxType{ssaop.OpcodeTable[v.Op].AuxType, ssaop.OpcodeTable[p.Op].AuxType} {
			case [2]ssaop.AuxType{ssaop.AuxTypeSymOff, ssaop.AuxTypeInt32}:
				// TODO: introduce auxSymOff32
				if !isInImmediateRange(v.AuxInt + p.AuxInt) {
					continue
				}
				v.AuxInt += p.AuxInt
			case [2]ssaop.AuxType{ssaop.AuxTypeSymOff, ssaop.AuxTypeSymOff}:
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
			case [2]ssaop.AuxType{ssaop.AuxTypeSymValAndOff, ssaop.AuxTypeInt32}:
				vo := ssacore.ValAndOff(v.AuxInt)
				if !vo.CanAdd64(p.AuxInt) {
					continue
				}
				v.AuxInt = int64(vo.AddOffset64(p.AuxInt))
			case [2]ssaop.AuxType{ssaop.AuxTypeSymValAndOff, ssaop.AuxTypeSymOff}:
				vo := ssacore.ValAndOff(v.AuxInt)
				if v.Aux != nil && p.Aux != nil {
					continue
				}
				if !vo.CanAdd64(p.AuxInt) {
					continue
				}
				if p.Aux != nil {
					v.Aux = p.Aux
				}
				v.AuxInt = int64(vo.AddOffset64(p.AuxInt))
			case [2]ssaop.AuxType{ssaop.AuxTypeSymOff, ssaop.AuxTypeNone}:
				// nothing to do
			case [2]ssaop.AuxType{ssaop.AuxTypeSymValAndOff, ssaop.AuxTypeNone}:
				// nothing to do
			default:
				f.Fatalf("unknown aux combining for %s and %s\n", v.Op, p.Op)
			}
			// Combine the operations.
			tmp = append(tmp[:0], v.Args[:ptrIndex]...)
			tmp = append(tmp, p.Args...)
			tmp = append(tmp, v.Args[ptrIndex+1:]...)
			v.ResetArgs()
			v.Op = c
			v.AddArgs(tmp...)
			if needSplit[c] {
				// It turns out that some of the combined instructions have faster two-instruction equivalents,
				// but not the two instructions that led to them being combined here.  For example
				// (CMPBconstload c (ADDQ x y)) -> (CMPBconstloadidx1 c x y) -> (CMPB c (MOVBloadidx1 x y))
				// The final pair of instructions turns out to be notably faster, at least in some benchmarks.
				f.Config.SplitLoad(v)
			}
		}
	}
}

// combineFirst contains ops which appear in combine as the
// first part of the key.
var combineFirst = map[ssaop.Op]bool{}

func init() {
	for k := range combine {
		combineFirst[k[0]] = true
	}
}

// needSplit contains instructions that should be postprocessed by splitLoad
// into a more-efficient two-instruction form.
var needSplit = map[ssaop.Op]bool{
	ssaop.OpAMD64CMPBloadidx1: true,
	ssaop.OpAMD64CMPWloadidx1: true,
	ssaop.OpAMD64CMPLloadidx1: true,
	ssaop.OpAMD64CMPQloadidx1: true,
	ssaop.OpAMD64CMPWloadidx2: true,
	ssaop.OpAMD64CMPLloadidx4: true,
	ssaop.OpAMD64CMPQloadidx8: true,

	ssaop.OpAMD64CMPBconstloadidx1: true,
	ssaop.OpAMD64CMPWconstloadidx1: true,
	ssaop.OpAMD64CMPLconstloadidx1: true,
	ssaop.OpAMD64CMPQconstloadidx1: true,
	ssaop.OpAMD64CMPWconstloadidx2: true,
	ssaop.OpAMD64CMPLconstloadidx4: true,
	ssaop.OpAMD64CMPQconstloadidx8: true,
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
var combine = map[[2]ssaop.Op]ssaop.Op{
	// amd64
	[2]ssaop.Op{ssaop.OpAMD64MOVBload, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64MOVBloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVWload, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64MOVWloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVLload, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64MOVLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVQload, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64MOVQloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVSSload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64MOVSSloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVSDload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64MOVSDloadidx1,

	[2]ssaop.Op{ssaop.OpAMD64MOVBstore, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64MOVBstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVWstore, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64MOVWstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVLstore, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64MOVLstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVQstore, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64MOVQstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVSSstore, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64MOVSSstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVSDstore, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64MOVSDstoreidx1,

	[2]ssaop.Op{ssaop.OpAMD64MOVBstoreconst, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64MOVBstoreconstidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVWstoreconst, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64MOVWstoreconstidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVLstoreconst, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64MOVLstoreconstidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVQstoreconst, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64MOVQstoreconstidx1,

	[2]ssaop.Op{ssaop.OpAMD64MOVBload, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64MOVBloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVWload, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64MOVWloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVWload, ssaop.OpAMD64LEAQ2}:  ssaop.OpAMD64MOVWloadidx2,
	[2]ssaop.Op{ssaop.OpAMD64MOVLload, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64MOVLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVLload, ssaop.OpAMD64LEAQ4}:  ssaop.OpAMD64MOVLloadidx4,
	[2]ssaop.Op{ssaop.OpAMD64MOVLload, ssaop.OpAMD64LEAQ8}:  ssaop.OpAMD64MOVLloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64MOVQload, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64MOVQloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVQload, ssaop.OpAMD64LEAQ8}:  ssaop.OpAMD64MOVQloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64MOVSSload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64MOVSSloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVSSload, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64MOVSSloadidx4,
	[2]ssaop.Op{ssaop.OpAMD64MOVSDload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64MOVSDloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVSDload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64MOVSDloadidx8,

	[2]ssaop.Op{ssaop.OpAMD64MOVBstore, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64MOVBstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVWstore, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64MOVWstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVWstore, ssaop.OpAMD64LEAQ2}:  ssaop.OpAMD64MOVWstoreidx2,
	[2]ssaop.Op{ssaop.OpAMD64MOVLstore, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64MOVLstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVLstore, ssaop.OpAMD64LEAQ4}:  ssaop.OpAMD64MOVLstoreidx4,
	[2]ssaop.Op{ssaop.OpAMD64MOVLstore, ssaop.OpAMD64LEAQ8}:  ssaop.OpAMD64MOVLstoreidx8,
	[2]ssaop.Op{ssaop.OpAMD64MOVQstore, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64MOVQstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVQstore, ssaop.OpAMD64LEAQ8}:  ssaop.OpAMD64MOVQstoreidx8,
	[2]ssaop.Op{ssaop.OpAMD64MOVSSstore, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64MOVSSstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVSSstore, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64MOVSSstoreidx4,
	[2]ssaop.Op{ssaop.OpAMD64MOVSDstore, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64MOVSDstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVSDstore, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64MOVSDstoreidx8,

	[2]ssaop.Op{ssaop.OpAMD64MOVBstoreconst, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64MOVBstoreconstidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVWstoreconst, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64MOVWstoreconstidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVWstoreconst, ssaop.OpAMD64LEAQ2}: ssaop.OpAMD64MOVWstoreconstidx2,
	[2]ssaop.Op{ssaop.OpAMD64MOVLstoreconst, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64MOVLstoreconstidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVLstoreconst, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64MOVLstoreconstidx4,
	[2]ssaop.Op{ssaop.OpAMD64MOVQstoreconst, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64MOVQstoreconstidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVQstoreconst, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64MOVQstoreconstidx8,

	[2]ssaop.Op{ssaop.OpAMD64SETEQstore, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64SETEQstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64SETNEstore, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64SETNEstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64SETLstore, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64SETLstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64SETLEstore, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64SETLEstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64SETGstore, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64SETGstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64SETGEstore, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64SETGEstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64SETBstore, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64SETBstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64SETBEstore, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64SETBEstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64SETAstore, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64SETAstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64SETAEstore, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64SETAEstoreidx1,

	// These instructions are re-split differently for performance, see needSplit above.
	// TODO if 386 versions are created, also update needSplit and _gen/386splitload.rules
	[2]ssaop.Op{ssaop.OpAMD64CMPBload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64CMPBloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64CMPWload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64CMPWloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64CMPLload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64CMPLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64CMPQload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64CMPQloadidx1,

	[2]ssaop.Op{ssaop.OpAMD64CMPBload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64CMPBloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64CMPWload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64CMPWloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64CMPWload, ssaop.OpAMD64LEAQ2}: ssaop.OpAMD64CMPWloadidx2,
	[2]ssaop.Op{ssaop.OpAMD64CMPLload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64CMPLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64CMPLload, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64CMPLloadidx4,
	[2]ssaop.Op{ssaop.OpAMD64CMPQload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64CMPQloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64CMPQload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64CMPQloadidx8,

	[2]ssaop.Op{ssaop.OpAMD64CMPBconstload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64CMPBconstloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64CMPWconstload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64CMPWconstloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64CMPLconstload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64CMPLconstloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64CMPQconstload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64CMPQconstloadidx1,

	[2]ssaop.Op{ssaop.OpAMD64CMPBconstload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64CMPBconstloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64CMPWconstload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64CMPWconstloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64CMPWconstload, ssaop.OpAMD64LEAQ2}: ssaop.OpAMD64CMPWconstloadidx2,
	[2]ssaop.Op{ssaop.OpAMD64CMPLconstload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64CMPLconstloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64CMPLconstload, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64CMPLconstloadidx4,
	[2]ssaop.Op{ssaop.OpAMD64CMPQconstload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64CMPQconstloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64CMPQconstload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64CMPQconstloadidx8,

	[2]ssaop.Op{ssaop.OpAMD64ADDLload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64ADDLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64ADDQload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64ADDQloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64SUBLload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64SUBLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64SUBQload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64SUBQloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64ANDLload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64ANDLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64ANDQload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64ANDQloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64ORLload, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64ORLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64ORQload, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64ORQloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64XORLload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64XORLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64XORQload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64XORQloadidx1,

	[2]ssaop.Op{ssaop.OpAMD64ADDLload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64ADDLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64ADDLload, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64ADDLloadidx4,
	[2]ssaop.Op{ssaop.OpAMD64ADDLload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64ADDLloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64ADDQload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64ADDQloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64ADDQload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64ADDQloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64SUBLload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64SUBLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64SUBLload, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64SUBLloadidx4,
	[2]ssaop.Op{ssaop.OpAMD64SUBLload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64SUBLloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64SUBQload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64SUBQloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64SUBQload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64SUBQloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64ANDLload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64ANDLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64ANDLload, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64ANDLloadidx4,
	[2]ssaop.Op{ssaop.OpAMD64ANDLload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64ANDLloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64ANDQload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64ANDQloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64ANDQload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64ANDQloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64ORLload, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64ORLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64ORLload, ssaop.OpAMD64LEAQ4}:  ssaop.OpAMD64ORLloadidx4,
	[2]ssaop.Op{ssaop.OpAMD64ORLload, ssaop.OpAMD64LEAQ8}:  ssaop.OpAMD64ORLloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64ORQload, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64ORQloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64ORQload, ssaop.OpAMD64LEAQ8}:  ssaop.OpAMD64ORQloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64XORLload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64XORLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64XORLload, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64XORLloadidx4,
	[2]ssaop.Op{ssaop.OpAMD64XORLload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64XORLloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64XORQload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64XORQloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64XORQload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64XORQloadidx8,

	[2]ssaop.Op{ssaop.OpAMD64ADDLmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64ADDLmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ADDQmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64ADDQmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64SUBLmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64SUBLmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64SUBQmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64SUBQmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ANDLmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64ANDLmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ANDQmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64ANDQmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ORLmodify, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64ORLmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ORQmodify, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64ORQmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64XORLmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64XORLmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64XORQmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64XORQmodifyidx1,

	[2]ssaop.Op{ssaop.OpAMD64ADDLmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64ADDLmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ADDLmodify, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64ADDLmodifyidx4,
	[2]ssaop.Op{ssaop.OpAMD64ADDLmodify, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64ADDLmodifyidx8,
	[2]ssaop.Op{ssaop.OpAMD64ADDQmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64ADDQmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ADDQmodify, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64ADDQmodifyidx8,
	[2]ssaop.Op{ssaop.OpAMD64SUBLmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64SUBLmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64SUBLmodify, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64SUBLmodifyidx4,
	[2]ssaop.Op{ssaop.OpAMD64SUBLmodify, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64SUBLmodifyidx8,
	[2]ssaop.Op{ssaop.OpAMD64SUBQmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64SUBQmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64SUBQmodify, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64SUBQmodifyidx8,
	[2]ssaop.Op{ssaop.OpAMD64ANDLmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64ANDLmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ANDLmodify, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64ANDLmodifyidx4,
	[2]ssaop.Op{ssaop.OpAMD64ANDLmodify, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64ANDLmodifyidx8,
	[2]ssaop.Op{ssaop.OpAMD64ANDQmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64ANDQmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ANDQmodify, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64ANDQmodifyidx8,
	[2]ssaop.Op{ssaop.OpAMD64ORLmodify, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64ORLmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ORLmodify, ssaop.OpAMD64LEAQ4}:  ssaop.OpAMD64ORLmodifyidx4,
	[2]ssaop.Op{ssaop.OpAMD64ORLmodify, ssaop.OpAMD64LEAQ8}:  ssaop.OpAMD64ORLmodifyidx8,
	[2]ssaop.Op{ssaop.OpAMD64ORQmodify, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64ORQmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ORQmodify, ssaop.OpAMD64LEAQ8}:  ssaop.OpAMD64ORQmodifyidx8,
	[2]ssaop.Op{ssaop.OpAMD64XORLmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64XORLmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64XORLmodify, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64XORLmodifyidx4,
	[2]ssaop.Op{ssaop.OpAMD64XORLmodify, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64XORLmodifyidx8,
	[2]ssaop.Op{ssaop.OpAMD64XORQmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64XORQmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64XORQmodify, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64XORQmodifyidx8,

	[2]ssaop.Op{ssaop.OpAMD64ADDLconstmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64ADDLconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ADDQconstmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64ADDQconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ANDLconstmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64ANDLconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ANDQconstmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64ANDQconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ORLconstmodify, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64ORLconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ORQconstmodify, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64ORQconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64XORLconstmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64XORLconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64XORQconstmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64XORQconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ADDWconstmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64ADDWconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ADDBconstmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64ADDBconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ANDWconstmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64ANDWconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ANDBconstmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64ANDBconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ORWconstmodify, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64ORWconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ORBconstmodify, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64ORBconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64XORWconstmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64XORWconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64XORBconstmodify, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64XORBconstmodifyidx1,

	[2]ssaop.Op{ssaop.OpAMD64ADDLconstmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64ADDLconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ADDLconstmodify, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64ADDLconstmodifyidx4,
	[2]ssaop.Op{ssaop.OpAMD64ADDLconstmodify, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64ADDLconstmodifyidx8,
	[2]ssaop.Op{ssaop.OpAMD64ADDQconstmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64ADDQconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ADDQconstmodify, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64ADDQconstmodifyidx8,
	[2]ssaop.Op{ssaop.OpAMD64ANDLconstmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64ANDLconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ANDLconstmodify, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64ANDLconstmodifyidx4,
	[2]ssaop.Op{ssaop.OpAMD64ANDLconstmodify, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64ANDLconstmodifyidx8,
	[2]ssaop.Op{ssaop.OpAMD64ANDQconstmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64ANDQconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ANDQconstmodify, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64ANDQconstmodifyidx8,
	[2]ssaop.Op{ssaop.OpAMD64ORLconstmodify, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64ORLconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ORLconstmodify, ssaop.OpAMD64LEAQ4}:  ssaop.OpAMD64ORLconstmodifyidx4,
	[2]ssaop.Op{ssaop.OpAMD64ORLconstmodify, ssaop.OpAMD64LEAQ8}:  ssaop.OpAMD64ORLconstmodifyidx8,
	[2]ssaop.Op{ssaop.OpAMD64ORQconstmodify, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64ORQconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ORQconstmodify, ssaop.OpAMD64LEAQ8}:  ssaop.OpAMD64ORQconstmodifyidx8,
	[2]ssaop.Op{ssaop.OpAMD64XORLconstmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64XORLconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64XORLconstmodify, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64XORLconstmodifyidx4,
	[2]ssaop.Op{ssaop.OpAMD64XORLconstmodify, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64XORLconstmodifyidx8,
	[2]ssaop.Op{ssaop.OpAMD64XORQconstmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64XORQconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64XORQconstmodify, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64XORQconstmodifyidx8,
	[2]ssaop.Op{ssaop.OpAMD64ADDWconstmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64ADDWconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ADDWconstmodify, ssaop.OpAMD64LEAQ2}: ssaop.OpAMD64ADDWconstmodifyidx2,
	[2]ssaop.Op{ssaop.OpAMD64ADDBconstmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64ADDBconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ANDWconstmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64ANDWconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ANDWconstmodify, ssaop.OpAMD64LEAQ2}: ssaop.OpAMD64ANDWconstmodifyidx2,
	[2]ssaop.Op{ssaop.OpAMD64ANDBconstmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64ANDBconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ORWconstmodify, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64ORWconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64ORWconstmodify, ssaop.OpAMD64LEAQ2}:  ssaop.OpAMD64ORWconstmodifyidx2,
	[2]ssaop.Op{ssaop.OpAMD64ORBconstmodify, ssaop.OpAMD64LEAQ1}:  ssaop.OpAMD64ORBconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64XORWconstmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64XORWconstmodifyidx1,
	[2]ssaop.Op{ssaop.OpAMD64XORWconstmodify, ssaop.OpAMD64LEAQ2}: ssaop.OpAMD64XORWconstmodifyidx2,
	[2]ssaop.Op{ssaop.OpAMD64XORBconstmodify, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64XORBconstmodifyidx1,

	[2]ssaop.Op{ssaop.OpAMD64ADDSSload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64ADDSSloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64ADDSSload, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64ADDSSloadidx4,
	[2]ssaop.Op{ssaop.OpAMD64ADDSDload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64ADDSDloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64ADDSDload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64ADDSDloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64SUBSSload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64SUBSSloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64SUBSSload, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64SUBSSloadidx4,
	[2]ssaop.Op{ssaop.OpAMD64SUBSDload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64SUBSDloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64SUBSDload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64SUBSDloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64MULSSload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64MULSSloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64MULSSload, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64MULSSloadidx4,
	[2]ssaop.Op{ssaop.OpAMD64MULSDload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64MULSDloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64MULSDload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64MULSDloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64DIVSSload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64DIVSSloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64DIVSSload, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64DIVSSloadidx4,
	[2]ssaop.Op{ssaop.OpAMD64DIVSDload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64DIVSDloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64DIVSDload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64DIVSDloadidx8,

	[2]ssaop.Op{ssaop.OpAMD64SARXLload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64SARXLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64SARXQload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64SARXQloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64SHLXLload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64SHLXLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64SHLXQload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64SHLXQloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64SHRXLload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64SHRXLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64SHRXQload, ssaop.OpAMD64ADDQ}: ssaop.OpAMD64SHRXQloadidx1,

	[2]ssaop.Op{ssaop.OpAMD64SARXLload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64SARXLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64SARXLload, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64SARXLloadidx4,
	[2]ssaop.Op{ssaop.OpAMD64SARXLload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64SARXLloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64SARXQload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64SARXQloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64SARXQload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64SARXQloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64SHLXLload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64SHLXLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64SHLXLload, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64SHLXLloadidx4,
	[2]ssaop.Op{ssaop.OpAMD64SHLXLload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64SHLXLloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64SHLXQload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64SHLXQloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64SHLXQload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64SHLXQloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64SHRXLload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64SHRXLloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64SHRXLload, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64SHRXLloadidx4,
	[2]ssaop.Op{ssaop.OpAMD64SHRXLload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64SHRXLloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64SHRXQload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64SHRXQloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64SHRXQload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64SHRXQloadidx8,

	// amd64/v3
	[2]ssaop.Op{ssaop.OpAMD64MOVBELload, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64MOVBELloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVBEQload, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64MOVBEQloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVBELload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64MOVBELloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVBELload, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64MOVBELloadidx4,
	[2]ssaop.Op{ssaop.OpAMD64MOVBELload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64MOVBELloadidx8,
	[2]ssaop.Op{ssaop.OpAMD64MOVBEQload, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64MOVBEQloadidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVBEQload, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64MOVBEQloadidx8,

	[2]ssaop.Op{ssaop.OpAMD64MOVBEWstore, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64MOVBEWstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVBELstore, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64MOVBELstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVBEQstore, ssaop.OpAMD64ADDQ}:  ssaop.OpAMD64MOVBEQstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVBEWstore, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64MOVBEWstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVBEWstore, ssaop.OpAMD64LEAQ2}: ssaop.OpAMD64MOVBEWstoreidx2,
	[2]ssaop.Op{ssaop.OpAMD64MOVBELstore, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64MOVBELstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVBELstore, ssaop.OpAMD64LEAQ4}: ssaop.OpAMD64MOVBELstoreidx4,
	[2]ssaop.Op{ssaop.OpAMD64MOVBELstore, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64MOVBELstoreidx8,
	[2]ssaop.Op{ssaop.OpAMD64MOVBEQstore, ssaop.OpAMD64LEAQ1}: ssaop.OpAMD64MOVBEQstoreidx1,
	[2]ssaop.Op{ssaop.OpAMD64MOVBEQstore, ssaop.OpAMD64LEAQ8}: ssaop.OpAMD64MOVBEQstoreidx8,

	// 386
	[2]ssaop.Op{ssaop.Op386MOVBload, ssaop.Op386ADDL}:  ssaop.Op386MOVBloadidx1,
	[2]ssaop.Op{ssaop.Op386MOVWload, ssaop.Op386ADDL}:  ssaop.Op386MOVWloadidx1,
	[2]ssaop.Op{ssaop.Op386MOVLload, ssaop.Op386ADDL}:  ssaop.Op386MOVLloadidx1,
	[2]ssaop.Op{ssaop.Op386MOVSSload, ssaop.Op386ADDL}: ssaop.Op386MOVSSloadidx1,
	[2]ssaop.Op{ssaop.Op386MOVSDload, ssaop.Op386ADDL}: ssaop.Op386MOVSDloadidx1,

	[2]ssaop.Op{ssaop.Op386MOVBstore, ssaop.Op386ADDL}:  ssaop.Op386MOVBstoreidx1,
	[2]ssaop.Op{ssaop.Op386MOVWstore, ssaop.Op386ADDL}:  ssaop.Op386MOVWstoreidx1,
	[2]ssaop.Op{ssaop.Op386MOVLstore, ssaop.Op386ADDL}:  ssaop.Op386MOVLstoreidx1,
	[2]ssaop.Op{ssaop.Op386MOVSSstore, ssaop.Op386ADDL}: ssaop.Op386MOVSSstoreidx1,
	[2]ssaop.Op{ssaop.Op386MOVSDstore, ssaop.Op386ADDL}: ssaop.Op386MOVSDstoreidx1,

	[2]ssaop.Op{ssaop.Op386MOVBstoreconst, ssaop.Op386ADDL}: ssaop.Op386MOVBstoreconstidx1,
	[2]ssaop.Op{ssaop.Op386MOVWstoreconst, ssaop.Op386ADDL}: ssaop.Op386MOVWstoreconstidx1,
	[2]ssaop.Op{ssaop.Op386MOVLstoreconst, ssaop.Op386ADDL}: ssaop.Op386MOVLstoreconstidx1,

	[2]ssaop.Op{ssaop.Op386MOVBload, ssaop.Op386LEAL1}:  ssaop.Op386MOVBloadidx1,
	[2]ssaop.Op{ssaop.Op386MOVWload, ssaop.Op386LEAL1}:  ssaop.Op386MOVWloadidx1,
	[2]ssaop.Op{ssaop.Op386MOVWload, ssaop.Op386LEAL2}:  ssaop.Op386MOVWloadidx2,
	[2]ssaop.Op{ssaop.Op386MOVLload, ssaop.Op386LEAL1}:  ssaop.Op386MOVLloadidx1,
	[2]ssaop.Op{ssaop.Op386MOVLload, ssaop.Op386LEAL4}:  ssaop.Op386MOVLloadidx4,
	[2]ssaop.Op{ssaop.Op386MOVSSload, ssaop.Op386LEAL1}: ssaop.Op386MOVSSloadidx1,
	[2]ssaop.Op{ssaop.Op386MOVSSload, ssaop.Op386LEAL4}: ssaop.Op386MOVSSloadidx4,
	[2]ssaop.Op{ssaop.Op386MOVSDload, ssaop.Op386LEAL1}: ssaop.Op386MOVSDloadidx1,
	[2]ssaop.Op{ssaop.Op386MOVSDload, ssaop.Op386LEAL8}: ssaop.Op386MOVSDloadidx8,

	[2]ssaop.Op{ssaop.Op386MOVBstore, ssaop.Op386LEAL1}:  ssaop.Op386MOVBstoreidx1,
	[2]ssaop.Op{ssaop.Op386MOVWstore, ssaop.Op386LEAL1}:  ssaop.Op386MOVWstoreidx1,
	[2]ssaop.Op{ssaop.Op386MOVWstore, ssaop.Op386LEAL2}:  ssaop.Op386MOVWstoreidx2,
	[2]ssaop.Op{ssaop.Op386MOVLstore, ssaop.Op386LEAL1}:  ssaop.Op386MOVLstoreidx1,
	[2]ssaop.Op{ssaop.Op386MOVLstore, ssaop.Op386LEAL4}:  ssaop.Op386MOVLstoreidx4,
	[2]ssaop.Op{ssaop.Op386MOVSSstore, ssaop.Op386LEAL1}: ssaop.Op386MOVSSstoreidx1,
	[2]ssaop.Op{ssaop.Op386MOVSSstore, ssaop.Op386LEAL4}: ssaop.Op386MOVSSstoreidx4,
	[2]ssaop.Op{ssaop.Op386MOVSDstore, ssaop.Op386LEAL1}: ssaop.Op386MOVSDstoreidx1,
	[2]ssaop.Op{ssaop.Op386MOVSDstore, ssaop.Op386LEAL8}: ssaop.Op386MOVSDstoreidx8,

	[2]ssaop.Op{ssaop.Op386MOVBstoreconst, ssaop.Op386LEAL1}: ssaop.Op386MOVBstoreconstidx1,
	[2]ssaop.Op{ssaop.Op386MOVWstoreconst, ssaop.Op386LEAL1}: ssaop.Op386MOVWstoreconstidx1,
	[2]ssaop.Op{ssaop.Op386MOVWstoreconst, ssaop.Op386LEAL2}: ssaop.Op386MOVWstoreconstidx2,
	[2]ssaop.Op{ssaop.Op386MOVLstoreconst, ssaop.Op386LEAL1}: ssaop.Op386MOVLstoreconstidx1,
	[2]ssaop.Op{ssaop.Op386MOVLstoreconst, ssaop.Op386LEAL4}: ssaop.Op386MOVLstoreconstidx4,

	[2]ssaop.Op{ssaop.Op386ADDLload, ssaop.Op386LEAL4}: ssaop.Op386ADDLloadidx4,
	[2]ssaop.Op{ssaop.Op386SUBLload, ssaop.Op386LEAL4}: ssaop.Op386SUBLloadidx4,
	[2]ssaop.Op{ssaop.Op386MULLload, ssaop.Op386LEAL4}: ssaop.Op386MULLloadidx4,
	[2]ssaop.Op{ssaop.Op386ANDLload, ssaop.Op386LEAL4}: ssaop.Op386ANDLloadidx4,
	[2]ssaop.Op{ssaop.Op386ORLload, ssaop.Op386LEAL4}:  ssaop.Op386ORLloadidx4,
	[2]ssaop.Op{ssaop.Op386XORLload, ssaop.Op386LEAL4}: ssaop.Op386XORLloadidx4,

	[2]ssaop.Op{ssaop.Op386ADDLmodify, ssaop.Op386LEAL4}: ssaop.Op386ADDLmodifyidx4,
	[2]ssaop.Op{ssaop.Op386SUBLmodify, ssaop.Op386LEAL4}: ssaop.Op386SUBLmodifyidx4,
	[2]ssaop.Op{ssaop.Op386ANDLmodify, ssaop.Op386LEAL4}: ssaop.Op386ANDLmodifyidx4,
	[2]ssaop.Op{ssaop.Op386ORLmodify, ssaop.Op386LEAL4}:  ssaop.Op386ORLmodifyidx4,
	[2]ssaop.Op{ssaop.Op386XORLmodify, ssaop.Op386LEAL4}: ssaop.Op386XORLmodifyidx4,

	[2]ssaop.Op{ssaop.Op386ADDLconstmodify, ssaop.Op386LEAL4}: ssaop.Op386ADDLconstmodifyidx4,
	[2]ssaop.Op{ssaop.Op386ANDLconstmodify, ssaop.Op386LEAL4}: ssaop.Op386ANDLconstmodifyidx4,
	[2]ssaop.Op{ssaop.Op386ORLconstmodify, ssaop.Op386LEAL4}:  ssaop.Op386ORLconstmodifyidx4,
	[2]ssaop.Op{ssaop.Op386XORLconstmodify, ssaop.Op386LEAL4}: ssaop.Op386XORLconstmodifyidx4,

	// s390x
	[2]ssaop.Op{ssaop.OpS390XMOVDload, ssaop.OpS390XADD}: ssaop.OpS390XMOVDloadidx,
	[2]ssaop.Op{ssaop.OpS390XMOVWload, ssaop.OpS390XADD}: ssaop.OpS390XMOVWloadidx,
	[2]ssaop.Op{ssaop.OpS390XMOVHload, ssaop.OpS390XADD}: ssaop.OpS390XMOVHloadidx,
	[2]ssaop.Op{ssaop.OpS390XMOVBload, ssaop.OpS390XADD}: ssaop.OpS390XMOVBloadidx,

	[2]ssaop.Op{ssaop.OpS390XMOVWZload, ssaop.OpS390XADD}: ssaop.OpS390XMOVWZloadidx,
	[2]ssaop.Op{ssaop.OpS390XMOVHZload, ssaop.OpS390XADD}: ssaop.OpS390XMOVHZloadidx,
	[2]ssaop.Op{ssaop.OpS390XMOVBZload, ssaop.OpS390XADD}: ssaop.OpS390XMOVBZloadidx,

	[2]ssaop.Op{ssaop.OpS390XMOVDBRload, ssaop.OpS390XADD}: ssaop.OpS390XMOVDBRloadidx,
	[2]ssaop.Op{ssaop.OpS390XMOVWBRload, ssaop.OpS390XADD}: ssaop.OpS390XMOVWBRloadidx,
	[2]ssaop.Op{ssaop.OpS390XMOVHBRload, ssaop.OpS390XADD}: ssaop.OpS390XMOVHBRloadidx,

	[2]ssaop.Op{ssaop.OpS390XFMOVDload, ssaop.OpS390XADD}: ssaop.OpS390XFMOVDloadidx,
	[2]ssaop.Op{ssaop.OpS390XFMOVSload, ssaop.OpS390XADD}: ssaop.OpS390XFMOVSloadidx,

	[2]ssaop.Op{ssaop.OpS390XMOVDstore, ssaop.OpS390XADD}: ssaop.OpS390XMOVDstoreidx,
	[2]ssaop.Op{ssaop.OpS390XMOVWstore, ssaop.OpS390XADD}: ssaop.OpS390XMOVWstoreidx,
	[2]ssaop.Op{ssaop.OpS390XMOVHstore, ssaop.OpS390XADD}: ssaop.OpS390XMOVHstoreidx,
	[2]ssaop.Op{ssaop.OpS390XMOVBstore, ssaop.OpS390XADD}: ssaop.OpS390XMOVBstoreidx,

	[2]ssaop.Op{ssaop.OpS390XMOVDBRstore, ssaop.OpS390XADD}: ssaop.OpS390XMOVDBRstoreidx,
	[2]ssaop.Op{ssaop.OpS390XMOVWBRstore, ssaop.OpS390XADD}: ssaop.OpS390XMOVWBRstoreidx,
	[2]ssaop.Op{ssaop.OpS390XMOVHBRstore, ssaop.OpS390XADD}: ssaop.OpS390XMOVHBRstoreidx,

	[2]ssaop.Op{ssaop.OpS390XFMOVDstore, ssaop.OpS390XADD}: ssaop.OpS390XFMOVDstoreidx,
	[2]ssaop.Op{ssaop.OpS390XFMOVSstore, ssaop.OpS390XADD}: ssaop.OpS390XFMOVSstoreidx,

	[2]ssaop.Op{ssaop.OpS390XMOVDload, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XMOVDloadidx,
	[2]ssaop.Op{ssaop.OpS390XMOVWload, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XMOVWloadidx,
	[2]ssaop.Op{ssaop.OpS390XMOVHload, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XMOVHloadidx,
	[2]ssaop.Op{ssaop.OpS390XMOVBload, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XMOVBloadidx,

	[2]ssaop.Op{ssaop.OpS390XMOVWZload, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XMOVWZloadidx,
	[2]ssaop.Op{ssaop.OpS390XMOVHZload, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XMOVHZloadidx,
	[2]ssaop.Op{ssaop.OpS390XMOVBZload, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XMOVBZloadidx,

	[2]ssaop.Op{ssaop.OpS390XMOVDBRload, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XMOVDBRloadidx,
	[2]ssaop.Op{ssaop.OpS390XMOVWBRload, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XMOVWBRloadidx,
	[2]ssaop.Op{ssaop.OpS390XMOVHBRload, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XMOVHBRloadidx,

	[2]ssaop.Op{ssaop.OpS390XFMOVDload, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XFMOVDloadidx,
	[2]ssaop.Op{ssaop.OpS390XFMOVSload, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XFMOVSloadidx,

	[2]ssaop.Op{ssaop.OpS390XMOVDstore, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XMOVDstoreidx,
	[2]ssaop.Op{ssaop.OpS390XMOVWstore, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XMOVWstoreidx,
	[2]ssaop.Op{ssaop.OpS390XMOVHstore, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XMOVHstoreidx,
	[2]ssaop.Op{ssaop.OpS390XMOVBstore, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XMOVBstoreidx,

	[2]ssaop.Op{ssaop.OpS390XMOVDBRstore, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XMOVDBRstoreidx,
	[2]ssaop.Op{ssaop.OpS390XMOVWBRstore, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XMOVWBRstoreidx,
	[2]ssaop.Op{ssaop.OpS390XMOVHBRstore, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XMOVHBRstoreidx,

	[2]ssaop.Op{ssaop.OpS390XFMOVDstore, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XFMOVDstoreidx,
	[2]ssaop.Op{ssaop.OpS390XFMOVSstore, ssaop.OpS390XMOVDaddridx}: ssaop.OpS390XFMOVSstoreidx,
}
