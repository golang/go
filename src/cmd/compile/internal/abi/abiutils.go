// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abi

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
	"fmt"
	"math"
	"sync"
)

//......................................................................
//
// Public/exported bits of the ABI utilities.
//

// ABIParamResultInfo stores the results of processing a given
// function type to compute stack layout and register assignments. For
// each input and output parameter we capture whether the param was
// register-assigned (and to which register(s)) or the stack offset
// for the param if is not going to be passed in registers according
// to the rules in the Go internal ABI specification (1.17).
type ABIParamResultInfo struct {
	inparams          []ABIParamAssignment // Includes receiver for method calls.  Does NOT include hidden closure pointer.
	outparams         []ABIParamAssignment
	offsetToSpillArea int64
	spillAreaSize     int64
	inRegistersUsed   int
	outRegistersUsed  int
	config            *ABIConfig // to enable String() method
}

func (a *ABIParamResultInfo) Config() *ABIConfig {
	return a.config
}

func (a *ABIParamResultInfo) InParams() []ABIParamAssignment {
	return a.inparams
}

func (a *ABIParamResultInfo) OutParams() []ABIParamAssignment {
	return a.outparams
}

func (a *ABIParamResultInfo) InRegistersUsed() int {
	return a.inRegistersUsed
}

func (a *ABIParamResultInfo) OutRegistersUsed() int {
	return a.outRegistersUsed
}

func (a *ABIParamResultInfo) InParam(i int) *ABIParamAssignment {
	return &a.inparams[i]
}

func (a *ABIParamResultInfo) OutParam(i int) *ABIParamAssignment {
	return &a.outparams[i]
}

func (a *ABIParamResultInfo) SpillAreaOffset() int64 {
	return a.offsetToSpillArea
}

func (a *ABIParamResultInfo) SpillAreaSize() int64 {
	return a.spillAreaSize
}

// ArgWidth returns the amount of stack needed for all the inputs
// and outputs of a function or method, including ABI-defined parameter
// slots and ABI-defined spill slots for register-resident parameters.
// The name is inherited from (*Type).ArgWidth(), which it replaces.
func (a *ABIParamResultInfo) ArgWidth() int64 {
	return a.spillAreaSize + a.offsetToSpillArea - a.config.LocalsOffset()
}

// RegIndex stores the index into the set of machine registers used by
// the ABI on a specific architecture for parameter passing.  RegIndex
// values 0 through N-1 (where N is the number of integer registers
// used for param passing according to the ABI rules) describe integer
// registers; values N through M (where M is the number of floating
// point registers used).  Thus if the ABI says there are 5 integer
// registers and 7 floating point registers, then RegIndex value of 4
// indicates the 5th integer register, and a RegIndex value of 11
// indicates the 7th floating point register.
type RegIndex uint8

// ABIParamAssignment holds information about how a specific param or
// result will be passed: in registers (in which case 'Registers' is
// populated) or on the stack (in which case 'Offset' is set to a
// non-negative stack offset). The values in 'Registers' are indices
// (as described above), not architected registers.
type ABIParamAssignment struct {
	Type      *types.Type
	Name      *ir.Name
	Registers []RegIndex
	offset    int32
}

// Offset returns the stack offset for addressing the parameter that "a" describes.
// This will panic if "a" describes a register-allocated parameter.
func (a *ABIParamAssignment) Offset() int32 {
	if len(a.Registers) > 0 {
		base.Fatalf("register allocated parameters have no offset")
	}
	return a.offset
}

// RegisterTypes returns a slice of the types of the registers
// corresponding to a slice of parameters.  The returned slice
// has capacity for one more, likely a memory type.
func RegisterTypes(apa []ABIParamAssignment) []*types.Type {
	rcount := 0
	for _, pa := range apa {
		rcount += len(pa.Registers)
	}
	if rcount == 0 {
		// Note that this catches top-level struct{} and [0]Foo, which are stack allocated.
		return make([]*types.Type, 0, 1)
	}
	rts := make([]*types.Type, 0, rcount+1)
	for _, pa := range apa {
		if len(pa.Registers) == 0 {
			continue
		}
		rts = appendParamTypes(rts, pa.Type)
	}
	return rts
}

func (pa *ABIParamAssignment) RegisterTypesAndOffsets() ([]*types.Type, []int64) {
	l := len(pa.Registers)
	if l == 0 {
		return nil, nil
	}
	typs := make([]*types.Type, 0, l)
	offs := make([]int64, 0, l)
	offs, _ = appendParamOffsets(offs, 0, pa.Type) // 0 is aligned for everything.
	return appendParamTypes(typs, pa.Type), offs
}

func appendParamTypes(rts []*types.Type, t *types.Type) []*types.Type {
	w := t.Size()
	if w == 0 {
		return rts
	}
	if t.IsScalar() || t.IsPtrShaped() {
		if t.IsComplex() {
			c := types.FloatForComplex(t)
			return append(rts, c, c)
		} else {
			if int(t.Size()) <= types.RegSize {
				return append(rts, t)
			}
			// assume 64bit int on 32-bit machine
			// TODO endianness? Should high-order (sign bits) word come first?
			if t.IsSigned() {
				rts = append(rts, types.Types[types.TINT32])
			} else {
				rts = append(rts, types.Types[types.TUINT32])
			}
			return append(rts, types.Types[types.TUINT32])
		}
	} else {
		typ := t.Kind()
		switch typ {
		case types.TARRAY:
			for i := int64(0); i < t.NumElem(); i++ { // 0 gets no registers, plus future-proofing.
				rts = appendParamTypes(rts, t.Elem())
			}
		case types.TSTRUCT:
			for _, f := range t.Fields() {
				if f.Type.Size() > 0 { // embedded zero-width types receive no registers
					rts = appendParamTypes(rts, f.Type)
				}
			}
		case types.TSLICE:
			return appendParamTypes(rts, synthSlice)
		case types.TSTRING:
			return appendParamTypes(rts, synthString)
		case types.TINTER:
			return appendParamTypes(rts, synthIface)
		}
	}
	return rts
}

// appendParamOffsets appends the offset(s) of type t, starting from "at",
// to input offsets, and returns the longer slice and the next unused offset.
// at should already be aligned for t.
func appendParamOffsets(offsets []int64, at int64, t *types.Type) ([]int64, int64) {
	w := t.Size()
	if w == 0 {
		return offsets, at
	}
	if t.IsScalar() || t.IsPtrShaped() {
		if t.IsComplex() || int(t.Size()) > types.RegSize { // complex and *int64 on 32-bit
			s := w / 2
			return append(offsets, at, at+s), at + w
		} else {
			return append(offsets, at), at + w
		}
	} else {
		typ := t.Kind()
		switch typ {
		case types.TARRAY:
			te := t.Elem()
			for i := int64(0); i < t.NumElem(); i++ {
				at = align(at, te)
				offsets, at = appendParamOffsets(offsets, at, te)
			}
		case types.TSTRUCT:
			at0 := at
			for i, f := range t.Fields() {
				at = at0 + f.Offset // Fields may be over-aligned, see wasm32.
				offsets, at = appendParamOffsets(offsets, at, f.Type)
				if f.Type.Size() == 0 && i == t.NumFields()-1 {
					at++ // last field has zero width
				}
			}
			at = align(at, t) // type size is rounded up to its alignment
		case types.TSLICE:
			return appendParamOffsets(offsets, at, synthSlice)
		case types.TSTRING:
			return appendParamOffsets(offsets, at, synthString)
		case types.TINTER:
			return appendParamOffsets(offsets, at, synthIface)
		}
	}
	return offsets, at
}

// FrameOffset returns the frame-pointer-relative location that a function
// would spill its input or output parameter to, if such a spill slot exists.
// If there is none defined (e.g., register-allocated outputs) it panics.
// For register-allocated inputs that is their spill offset reserved for morestack;
// for stack-allocated inputs and outputs, that is their location on the stack.
// (In a future version of the ABI, register-resident inputs may lose their defined
// spill area to help reduce stack sizes.)
func (a *ABIParamAssignment) FrameOffset(i *ABIParamResultInfo) int64 {
	if a.offset == -1 {
		base.Fatalf("function parameter has no ABI-defined frame-pointer offset")
	}
	if len(a.Registers) == 0 { // passed on stack
		return int64(a.offset) - i.config.LocalsOffset()
	}
	// spill area for registers
	return int64(a.offset) + i.SpillAreaOffset() - i.config.LocalsOffset()
}

// RegAmounts holds a specified number of integer/float registers.
type RegAmounts struct {
	intRegs   int
	floatRegs int
}

// ABIConfig captures the number of registers made available
// by the ABI rules for parameter passing and result returning.
type ABIConfig struct {
	// Do we need anything more than this?
	offsetForLocals int64 // e.g., obj.(*Link).Arch.FixedFrameSize -- extra linkage information on some architectures.
	regAmounts      RegAmounts
	which           obj.ABI
}

// NewABIConfig returns a new ABI configuration for an architecture with
// iRegsCount integer/pointer registers and fRegsCount floating point registers.
func NewABIConfig(iRegsCount, fRegsCount int, offsetForLocals int64, which uint8) *ABIConfig {
	return &ABIConfig{offsetForLocals: offsetForLocals, regAmounts: RegAmounts{iRegsCount, fRegsCount}, which: obj.ABI(which)}
}

// Copy returns config.
//
// TODO(mdempsky): Remove.
func (config *ABIConfig) Copy() *ABIConfig {
	return config
}

// Which returns the ABI number
func (config *ABIConfig) Which() obj.ABI {
	return config.which
}

// LocalsOffset returns the architecture-dependent offset from SP for args and results.
// In theory this is only used for debugging; it ought to already be incorporated into
// results from the ABI-related methods
func (config *ABIConfig) LocalsOffset() int64 {
	return config.offsetForLocals
}

// FloatIndexFor translates r into an index in the floating point parameter
// registers.  If the result is negative, the input index was actually for the
// integer parameter registers.
func (config *ABIConfig) FloatIndexFor(r RegIndex) int64 {
	return int64(r) - int64(config.regAmounts.intRegs)
}

// NumParamRegs returns the total number of registers used to
// represent a parameter of the given type, which must be register
// assignable.
func (config *ABIConfig) NumParamRegs(typ *types.Type) int {
	intRegs, floatRegs := typ.Registers()
	if intRegs == math.MaxUint8 && floatRegs == math.MaxUint8 {
		base.Fatalf("cannot represent parameters of type %v in registers", typ)
	}
	return int(intRegs) + int(floatRegs)
}

// ABIAnalyzeTypes takes slices of parameter and result types, and returns an ABIParamResultInfo,
// based on the given configuration.  This is the same result computed by config.ABIAnalyze applied to the
// corresponding method/function type, except that all the embedded parameter names are nil.
// This is intended for use by ssagen/ssa.go:(*state).rtcall, for runtime functions that lack a parsed function type.
func (config *ABIConfig) ABIAnalyzeTypes(params, results []*types.Type) *ABIParamResultInfo {
	setup()
	s := assignState{
		stackOffset: config.offsetForLocals,
		rTotal:      config.regAmounts,
	}

	assignParams := func(params []*types.Type, isResult bool) []ABIParamAssignment {
		res := make([]ABIParamAssignment, len(params))
		for i, param := range params {
			res[i] = s.assignParam(param, nil, isResult)
		}
		return res
	}

	info := &ABIParamResultInfo{config: config}

	// Inputs
	info.inparams = assignParams(params, false)
	s.stackOffset = types.RoundUp(s.stackOffset, int64(types.RegSize))
	info.inRegistersUsed = s.rUsed.intRegs + s.rUsed.floatRegs

	// Outputs
	s.rUsed = RegAmounts{}
	info.outparams = assignParams(results, true)
	// The spill area is at a register-aligned offset and its size is rounded up to a register alignment.
	// TODO in theory could align offset only to minimum required by spilled data types.
	info.offsetToSpillArea = alignTo(s.stackOffset, types.RegSize)
	info.spillAreaSize = alignTo(s.spillOffset, types.RegSize)
	info.outRegistersUsed = s.rUsed.intRegs + s.rUsed.floatRegs

	return info
}

// ABIAnalyzeFuncType takes a function type 'ft' and an ABI rules description
// 'config' and analyzes the function to determine how its parameters
// and results will be passed (in registers or on the stack), returning
// an ABIParamResultInfo object that holds the results of the analysis.
func (config *ABIConfig) ABIAnalyzeFuncType(ft *types.Type) *ABIParamResultInfo {
	setup()
	s := assignState{
		stackOffset: config.offsetForLocals,
		rTotal:      config.regAmounts,
	}

	assignParams := func(params []*types.Field, isResult bool) []ABIParamAssignment {
		res := make([]ABIParamAssignment, len(params))
		for i, param := range params {
			var name *ir.Name
			if param.Nname != nil {
				name = param.Nname.(*ir.Name)
			}
			res[i] = s.assignParam(param.Type, name, isResult)
		}
		return res
	}

	info := &ABIParamResultInfo{config: config}

	// Inputs
	info.inparams = assignParams(ft.RecvParams(), false)
	s.stackOffset = types.RoundUp(s.stackOffset, int64(types.RegSize))
	info.inRegistersUsed = s.rUsed.intRegs + s.rUsed.floatRegs

	// Outputs
	s.rUsed = RegAmounts{}
	info.outparams = assignParams(ft.Results(), true)
	// The spill area is at a register-aligned offset and its size is rounded up to a register alignment.
	// TODO in theory could align offset only to minimum required by spilled data types.
	info.offsetToSpillArea = alignTo(s.stackOffset, types.RegSize)
	info.spillAreaSize = alignTo(s.spillOffset, types.RegSize)
	info.outRegistersUsed = s.rUsed.intRegs + s.rUsed.floatRegs
	return info
}

// ABIAnalyze returns the same result as ABIAnalyzeFuncType, but also
// updates the offsets of all the receiver, input, and output fields.
// If setNname is true, it also sets the FrameOffset of the Nname for
// the field(s); this is for use when compiling a function and figuring out
// spill locations.  Doing this for callers can cause races for register
// outputs because their frame location transitions from BOGUS_FUNARG_OFFSET
// to zero to an as-if-AUTO offset that has no use for callers.
func (config *ABIConfig) ABIAnalyze(t *types.Type, setNname bool) *ABIParamResultInfo {
	result := config.ABIAnalyzeFuncType(t)

	// Fill in the frame offsets for receiver, inputs, results
	for i, f := range t.RecvParams() {
		config.updateOffset(result, f, result.inparams[i], false, setNname)
	}
	for i, f := range t.Results() {
		config.updateOffset(result, f, result.outparams[i], true, setNname)
	}
	return result
}

func (config *ABIConfig) updateOffset(result *ABIParamResultInfo, f *types.Field, a ABIParamAssignment, isResult, setNname bool) {
	if f.Offset != types.BADWIDTH {
		base.Fatalf("field offset for %s at %s has been set to %d", f.Sym, base.FmtPos(f.Pos), f.Offset)
	}

	// Everything except return values in registers has either a frame home (if not in a register) or a frame spill location.
	if !isResult || len(a.Registers) == 0 {
		// The type frame offset DOES NOT show effects of minimum frame size.
		// Getting this wrong breaks stackmaps, see liveness/plive.go:WriteFuncMap and typebits/typebits.go:Set
		off := a.FrameOffset(result)
		if setNname && f.Nname != nil {
			f.Nname.(*ir.Name).SetFrameOffset(off)
			f.Nname.(*ir.Name).SetIsOutputParamInRegisters(false)
		}
	} else {
		if setNname && f.Nname != nil {
			fname := f.Nname.(*ir.Name)
			fname.SetIsOutputParamInRegisters(true)
			fname.SetFrameOffset(0)
		}
	}
}

//......................................................................
//
// Non-public portions.

// regString produces a human-readable version of a RegIndex.
func (c *RegAmounts) regString(r RegIndex) string {
	if int(r) < c.intRegs {
		return fmt.Sprintf("I%d", int(r))
	} else if int(r) < c.intRegs+c.floatRegs {
		return fmt.Sprintf("F%d", int(r)-c.intRegs)
	}
	return fmt.Sprintf("<?>%d", r)
}

// ToString method renders an ABIParamAssignment in human-readable
// form, suitable for debugging or unit testing.
func (ri *ABIParamAssignment) ToString(config *ABIConfig, extra bool) string {
	regs := "R{"
	offname := "spilloffset" // offset is for spill for register(s)
	if len(ri.Registers) == 0 {
		offname = "offset" // offset is for memory arg
	}
	for _, r := range ri.Registers {
		regs += " " + config.regAmounts.regString(r)
		if extra {
			regs += fmt.Sprintf("(%d)", r)
		}
	}
	if extra {
		regs += fmt.Sprintf(" | #I=%d, #F=%d", config.regAmounts.intRegs, config.regAmounts.floatRegs)
	}
	return fmt.Sprintf("%s } %s: %d typ: %v", regs, offname, ri.offset, ri.Type)
}

// String method renders an ABIParamResultInfo in human-readable
// form, suitable for debugging or unit testing.
func (ri *ABIParamResultInfo) String() string {
	res := ""
	for k, p := range ri.inparams {
		res += fmt.Sprintf("IN %d: %s\n", k, p.ToString(ri.config, false))
	}
	for k, r := range ri.outparams {
		res += fmt.Sprintf("OUT %d: %s\n", k, r.ToString(ri.config, false))
	}
	res += fmt.Sprintf("offsetToSpillArea: %d spillAreaSize: %d",
		ri.offsetToSpillArea, ri.spillAreaSize)
	return res
}

// assignState holds intermediate state during the register assigning process
// for a given function signature.
type assignState struct {
	rTotal      RegAmounts // total reg amounts from ABI rules
	rUsed       RegAmounts // regs used by params completely assigned so far
	stackOffset int64      // current stack offset
	spillOffset int64      // current spill offset
}

// align returns a rounded up to t's alignment.
func align(a int64, t *types.Type) int64 {
	return alignTo(a, int(uint8(t.Alignment())))
}

// alignTo returns a rounded up to t, where t must be 0 or a power of 2.
func alignTo(a int64, t int) int64 {
	if t == 0 {
		return a
	}
	return types.RoundUp(a, int64(t))
}

// nextSlot allocates the next available slot for typ.
func nextSlot(offsetp *int64, typ *types.Type) int64 {
	offset := align(*offsetp, typ)
	*offsetp = offset + typ.Size()
	return offset
}

// allocateRegs returns an ordered list of register indices for a parameter or result
// that we've just determined to be register-assignable. The number of registers
// needed is assumed to be stored in state.pUsed.
func (state *assignState) allocateRegs(regs []RegIndex, t *types.Type) []RegIndex {
	if t.Size() == 0 {
		return regs
	}
	ri := state.rUsed.intRegs
	rf := state.rUsed.floatRegs
	if t.IsScalar() || t.IsPtrShaped() {
		if t.IsComplex() {
			regs = append(regs, RegIndex(rf+state.rTotal.intRegs), RegIndex(rf+1+state.rTotal.intRegs))
			rf += 2
		} else if t.IsFloat() {
			regs = append(regs, RegIndex(rf+state.rTotal.intRegs))
			rf += 1
		} else {
			n := (int(t.Size()) + types.RegSize - 1) / types.RegSize
			for i := 0; i < n; i++ { // looking ahead to really big integers
				regs = append(regs, RegIndex(ri))
				ri += 1
			}
		}
		state.rUsed.intRegs = ri
		state.rUsed.floatRegs = rf
		return regs
	} else {
		typ := t.Kind()
		switch typ {
		case types.TARRAY:
			for i := int64(0); i < t.NumElem(); i++ {
				regs = state.allocateRegs(regs, t.Elem())
			}
			return regs
		case types.TSTRUCT:
			for _, f := range t.Fields() {
				regs = state.allocateRegs(regs, f.Type)
			}
			return regs
		case types.TSLICE:
			return state.allocateRegs(regs, synthSlice)
		case types.TSTRING:
			return state.allocateRegs(regs, synthString)
		case types.TINTER:
			return state.allocateRegs(regs, synthIface)
		}
	}
	base.Fatalf("was not expecting type %s", t)
	panic("unreachable")
}

// synthOnce ensures that we only create the synth* fake types once.
var synthOnce sync.Once

// synthSlice, synthString, and syncIface are synthesized struct types
// meant to capture the underlying implementations of string/slice/interface.
var synthSlice *types.Type
var synthString *types.Type
var synthIface *types.Type

// setup performs setup for the register assignment utilities, manufacturing
// a small set of synthesized types that we'll need along the way.
func setup() {
	synthOnce.Do(func() {
		fname := types.BuiltinPkg.Lookup
		nxp := src.NoXPos
		bp := types.NewPtr(types.Types[types.TUINT8])
		it := types.Types[types.TINT]
		synthSlice = types.NewStruct([]*types.Field{
			types.NewField(nxp, fname("ptr"), bp),
			types.NewField(nxp, fname("len"), it),
			types.NewField(nxp, fname("cap"), it),
		})
		types.CalcStructSize(synthSlice)
		synthString = types.NewStruct([]*types.Field{
			types.NewField(nxp, fname("data"), bp),
			types.NewField(nxp, fname("len"), it),
		})
		types.CalcStructSize(synthString)
		unsp := types.Types[types.TUNSAFEPTR]
		synthIface = types.NewStruct([]*types.Field{
			types.NewField(nxp, fname("f1"), unsp),
			types.NewField(nxp, fname("f2"), unsp),
		})
		types.CalcStructSize(synthIface)
	})
}

// assignParam processes a given receiver, param, or result
// of field f to determine whether it can be register assigned.
// The result of the analysis is recorded in the result
// ABIParamResultInfo held in 'state'.
func (state *assignState) assignParam(typ *types.Type, name *ir.Name, isResult bool) ABIParamAssignment {
	registers := state.tryAllocRegs(typ)

	var offset int64 = -1
	if registers == nil { // stack allocated; needs stack slot
		offset = nextSlot(&state.stackOffset, typ)
	} else if !isResult { // register-allocated param; needs spill slot
		offset = nextSlot(&state.spillOffset, typ)
	}

	return ABIParamAssignment{
		Type:      typ,
		Name:      name,
		Registers: registers,
		offset:    int32(offset),
	}
}

// tryAllocRegs attempts to allocate registers to represent a
// parameter of the given type. If unsuccessful, it returns nil.
func (state *assignState) tryAllocRegs(typ *types.Type) []RegIndex {
	if typ.Size() == 0 {
		return nil // zero-size parameters are defined as being stack allocated
	}

	intRegs, floatRegs := typ.Registers()
	if int(intRegs) > state.rTotal.intRegs-state.rUsed.intRegs || int(floatRegs) > state.rTotal.floatRegs-state.rUsed.floatRegs {
		return nil // too few available registers
	}

	regs := make([]RegIndex, 0, int(intRegs)+int(floatRegs))
	return state.allocateRegs(regs, typ)
}

// ComputePadding returns a list of "post element" padding values in
// the case where we have a structure being passed in registers. Given
// a param assignment corresponding to a struct, it returns a list
// containing padding values for each field, e.g. the Kth element in
// the list is the amount of padding between field K and the following
// field. For things that are not structs (or structs without padding)
// it returns a list of zeros. Example:
//
//	type small struct {
//		x uint16
//		y uint8
//		z int32
//		w int32
//	}
//
// For this struct we would return a list [0, 1, 0, 0], meaning that
// we have one byte of padding after the second field, and no bytes of
// padding after any of the other fields. Input parameter "storage" is
// a slice with enough capacity to accommodate padding elements for
// the architected register set in question.
func (pa *ABIParamAssignment) ComputePadding(storage []uint64) []uint64 {
	nr := len(pa.Registers)
	padding := storage[:nr]
	for i := 0; i < nr; i++ {
		padding[i] = 0
	}
	if pa.Type.Kind() != types.TSTRUCT || nr == 0 {
		return padding
	}
	types := make([]*types.Type, 0, nr)
	types = appendParamTypes(types, pa.Type)
	if len(types) != nr {
		panic("internal error")
	}
	offsets, _ := appendParamOffsets([]int64{}, 0, pa.Type)
	off := int64(0)
	for idx, t := range types {
		ts := t.Size()
		off += int64(ts)
		if idx < len(types)-1 {
			noff := offsets[idx+1]
			if noff != off {
				p := noff - off
				padding[idx] = uint64(p)
				off += p
			}
		}
	}
	return padding
}
