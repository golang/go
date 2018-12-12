// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// When breaking up a combined load-compare to separated load and compare operations,
// opLoad specifies the load operation, and opCmp specifies the compare operation.
type typeCmdLoadMap struct {
	opLoad Op
	opCmp  Op
}

var opCmpLoadMap = map[Op]typeCmdLoadMap{
	OpAMD64CMPQload:      {OpAMD64MOVQload, OpAMD64CMPQ},
	OpAMD64CMPLload:      {OpAMD64MOVLload, OpAMD64CMPL},
	OpAMD64CMPWload:      {OpAMD64MOVWload, OpAMD64CMPW},
	OpAMD64CMPBload:      {OpAMD64MOVBload, OpAMD64CMPB},
	Op386CMPLload:        {Op386MOVLload, Op386CMPL},
	Op386CMPWload:        {Op386MOVWload, Op386CMPW},
	Op386CMPBload:        {Op386MOVBload, Op386CMPB},
	OpAMD64CMPQconstload: {OpAMD64MOVQload, OpAMD64CMPQconst},
	OpAMD64CMPLconstload: {OpAMD64MOVLload, OpAMD64CMPLconst},
	OpAMD64CMPWconstload: {OpAMD64MOVWload, OpAMD64CMPWconst},
	OpAMD64CMPBconstload: {OpAMD64MOVBload, OpAMD64CMPBconst},
	Op386CMPLconstload:   {Op386MOVLload, Op386CMPLconst},
	Op386CMPWconstload:   {Op386MOVWload, Op386CMPWconst},
	Op386CMPBconstload:   {Op386MOVBload, Op386CMPBconst},
}

// flagalloc allocates the flag register among all the flag-generating
// instructions. Flag values are recomputed if they need to be
// spilled/restored.
func flagalloc(f *Func) {
	// Compute the in-register flag value we want at the end of
	// each block. This is basically a best-effort live variable
	// analysis, so it can be much simpler than a full analysis.
	end := make([]*Value, f.NumBlocks())
	po := f.postorder()
	for n := 0; n < 2; n++ {
		for _, b := range po {
			// Walk values backwards to figure out what flag
			// value we want in the flag register at the start
			// of the block.
			flag := end[b.ID]
			if b.Control != nil && b.Control.Type.IsFlags() {
				flag = b.Control
			}
			for j := len(b.Values) - 1; j >= 0; j-- {
				v := b.Values[j]
				if v == flag {
					flag = nil
				}
				if v.clobbersFlags() {
					flag = nil
				}
				for _, a := range v.Args {
					if a.Type.IsFlags() {
						flag = a
					}
				}
			}
			if flag != nil {
				for _, e := range b.Preds {
					p := e.b
					end[p.ID] = flag
				}
			}
		}
	}

	// For blocks which have a flags control value, that's the only value
	// we can leave in the flags register at the end of the block. (There
	// is no place to put a flag regeneration instruction.)
	for _, b := range f.Blocks {
		v := b.Control
		if v != nil && v.Type.IsFlags() && end[b.ID] != v {
			end[b.ID] = nil
		}
		if b.Kind == BlockDefer {
			// Defer blocks internally use/clobber the flags value.
			end[b.ID] = nil
		}
	}

	// Compute which flags values will need to be spilled.
	spill := map[ID]bool{}
	for _, b := range f.Blocks {
		var flag *Value
		if len(b.Preds) > 0 {
			flag = end[b.Preds[0].b.ID]
		}
		for _, v := range b.Values {
			for _, a := range v.Args {
				if !a.Type.IsFlags() {
					continue
				}
				if a == flag {
					continue
				}
				// a will need to be restored here.
				spill[a.ID] = true
				flag = a
			}
			if v.clobbersFlags() {
				flag = nil
			}
			if v.Type.IsFlags() {
				flag = v
			}
		}
		if v := b.Control; v != nil && v != flag && v.Type.IsFlags() {
			spill[v.ID] = true
		}
		if v := end[b.ID]; v != nil && v != flag {
			spill[v.ID] = true
		}
	}

	// Add flag spill and recomputation where they are needed.
	// TODO: Remove original instructions if they are never used.
	var oldSched []*Value
	for _, b := range f.Blocks {
		oldSched = append(oldSched[:0], b.Values...)
		b.Values = b.Values[:0]
		// The current live flag value (the pre-flagalloc copy).
		var flag *Value
		if len(b.Preds) > 0 {
			flag = end[b.Preds[0].b.ID]
			// Note: the following condition depends on the lack of critical edges.
			for _, e := range b.Preds[1:] {
				p := e.b
				if end[p.ID] != flag {
					f.Fatalf("live flag in %s's predecessors not consistent", b)
				}
			}
		}
		for _, v := range oldSched {
			if v.Op == OpPhi && v.Type.IsFlags() {
				f.Fatalf("phi of flags not supported: %s", v.LongString())
			}

			// If v will be spilled, and v uses memory, then we must split it
			// into a load + a flag generator.
			// TODO: figure out how to do this without arch-dependent code.
			if spill[v.ID] && v.MemoryArg() != nil {
				switch v.Op {
				case OpAMD64CMPQload:
					load := b.NewValue2IA(v.Pos, opCmpLoadMap[v.Op].opLoad, f.Config.Types.UInt64, v.AuxInt, v.Aux, v.Args[0], v.Args[2])
					v.Op = opCmpLoadMap[v.Op].opCmp
					v.AuxInt = 0
					v.Aux = nil
					v.SetArgs2(load, v.Args[1])
				case OpAMD64CMPLload, Op386CMPLload:
					load := b.NewValue2IA(v.Pos, opCmpLoadMap[v.Op].opLoad, f.Config.Types.UInt32, v.AuxInt, v.Aux, v.Args[0], v.Args[2])
					v.Op = opCmpLoadMap[v.Op].opCmp
					v.AuxInt = 0
					v.Aux = nil
					v.SetArgs2(load, v.Args[1])
				case OpAMD64CMPWload, Op386CMPWload:
					load := b.NewValue2IA(v.Pos, opCmpLoadMap[v.Op].opLoad, f.Config.Types.UInt16, v.AuxInt, v.Aux, v.Args[0], v.Args[2])
					v.Op = opCmpLoadMap[v.Op].opCmp
					v.AuxInt = 0
					v.Aux = nil
					v.SetArgs2(load, v.Args[1])
				case OpAMD64CMPBload, Op386CMPBload:
					load := b.NewValue2IA(v.Pos, opCmpLoadMap[v.Op].opLoad, f.Config.Types.UInt8, v.AuxInt, v.Aux, v.Args[0], v.Args[2])
					v.Op = opCmpLoadMap[v.Op].opCmp
					v.AuxInt = 0
					v.Aux = nil
					v.SetArgs2(load, v.Args[1])

				case OpAMD64CMPQconstload:
					vo := v.AuxValAndOff()
					load := b.NewValue2IA(v.Pos, opCmpLoadMap[v.Op].opLoad, f.Config.Types.UInt64, vo.Off(), v.Aux, v.Args[0], v.Args[1])
					v.Op = opCmpLoadMap[v.Op].opCmp
					v.AuxInt = vo.Val()
					v.Aux = nil
					v.SetArgs1(load)
				case OpAMD64CMPLconstload, Op386CMPLconstload:
					vo := v.AuxValAndOff()
					load := b.NewValue2IA(v.Pos, opCmpLoadMap[v.Op].opLoad, f.Config.Types.UInt32, vo.Off(), v.Aux, v.Args[0], v.Args[1])
					v.Op = opCmpLoadMap[v.Op].opCmp
					v.AuxInt = vo.Val()
					v.Aux = nil
					v.SetArgs1(load)
				case OpAMD64CMPWconstload, Op386CMPWconstload:
					vo := v.AuxValAndOff()
					load := b.NewValue2IA(v.Pos, opCmpLoadMap[v.Op].opLoad, f.Config.Types.UInt16, vo.Off(), v.Aux, v.Args[0], v.Args[1])
					v.Op = opCmpLoadMap[v.Op].opCmp
					v.AuxInt = vo.Val()
					v.Aux = nil
					v.SetArgs1(load)
				case OpAMD64CMPBconstload, Op386CMPBconstload:
					vo := v.AuxValAndOff()
					load := b.NewValue2IA(v.Pos, opCmpLoadMap[v.Op].opLoad, f.Config.Types.UInt8, vo.Off(), v.Aux, v.Args[0], v.Args[1])
					v.Op = opCmpLoadMap[v.Op].opCmp
					v.AuxInt = vo.Val()
					v.Aux = nil
					v.SetArgs1(load)

				default:
					f.Fatalf("can't split flag generator: %s", v.LongString())
				}

			}

			// Make sure any flag arg of v is in the flags register.
			// If not, recompute it.
			for i, a := range v.Args {
				if !a.Type.IsFlags() {
					continue
				}
				if a == flag {
					continue
				}
				// Recalculate a
				c := copyFlags(a, b)
				// Update v.
				v.SetArg(i, c)
				// Remember the most-recently computed flag value.
				flag = a
			}
			// Issue v.
			b.Values = append(b.Values, v)
			if v.clobbersFlags() {
				flag = nil
			}
			if v.Type.IsFlags() {
				flag = v
			}
		}
		if v := b.Control; v != nil && v != flag && v.Type.IsFlags() {
			// Recalculate control value.
			c := copyFlags(v, b)
			b.SetControl(c)
			flag = v
		}
		if v := end[b.ID]; v != nil && v != flag {
			// Need to reissue flag generator for use by
			// subsequent blocks.
			copyFlags(v, b)
			// Note: this flag generator is not properly linked up
			// with the flag users. This breaks the SSA representation.
			// We could fix up the users with another pass, but for now
			// we'll just leave it.  (Regalloc has the same issue for
			// standard regs, and it runs next.)
		}
	}

	// Save live flag state for later.
	for _, b := range f.Blocks {
		b.FlagsLiveAtEnd = end[b.ID] != nil
	}
}

func (v *Value) clobbersFlags() bool {
	if opcodeTable[v.Op].clobberFlags {
		return true
	}
	if v.Type.IsTuple() && (v.Type.FieldType(0).IsFlags() || v.Type.FieldType(1).IsFlags()) {
		// This case handles the possibility where a flag value is generated but never used.
		// In that case, there's no corresponding Select to overwrite the flags value,
		// so we must consider flags clobbered by the tuple-generating instruction.
		return true
	}
	return false
}

// copyFlags copies v (flag generator) into b, returns the copy.
// If v's arg is also flags, copy recursively.
func copyFlags(v *Value, b *Block) *Value {
	flagsArgs := make(map[int]*Value)
	for i, a := range v.Args {
		if a.Type.IsFlags() || a.Type.IsTuple() {
			flagsArgs[i] = copyFlags(a, b)
		}
	}
	c := v.copyInto(b)
	for i, a := range flagsArgs {
		c.SetArg(i, a)
	}
	return c
}
