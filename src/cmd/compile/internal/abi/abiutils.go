// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abi

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
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
// non-negative stack offset. The values in 'Registers' are indices
// (as described above), not architected registers.
type ABIParamAssignment struct {
	Type      *types.Type
	Name      types.Object // should always be *ir.Name, used to match with a particular ssa.OpArg.
	Registers []RegIndex
	offset    int32
}

// Offset returns the stack offset for addressing the parameter that "a" describes.
// This will panic if "a" describes a register-allocated parameter.
func (a *ABIParamAssignment) Offset() int32 {
	if len(a.Registers) > 0 {
		panic("Register allocated parameters have no offset")
	}
	return a.offset
}

// SpillOffset returns the offset *within the spill area* for the parameter that "a" describes.
// Registers will be spilled here; if a memory home is needed (for a pointer method e.g.)
// then that will be the address.
// This will panic if "a" describes a stack-allocated parameter.
func (a *ABIParamAssignment) SpillOffset() int32 {
	if len(a.Registers) == 0 {
		panic("Stack-allocated parameters have no spill offset")
	}
	return a.offset
}

// FrameOffset returns the location that a value would spill to, if any exists.
// For register-allocated inputs, that is their spill offset reserved for morestack
// (might as well use it, it is there); for stack-allocated inputs and outputs,
// that is their location on the stack.  For register-allocated outputs, there is
// no defined spill area, so return -1.
func (a *ABIParamAssignment) FrameOffset(i *ABIParamResultInfo) int64 {
	if len(a.Registers) == 0 || a.offset == -1 {
		return int64(a.offset)
	}
	return int64(a.offset) + i.SpillAreaOffset()
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
	offsetForLocals  int64 // e.g., obj.(*Link).FixedFrameSize() -- extra linkage information on some architectures.
	regAmounts       RegAmounts
	regsForTypeCache map[*types.Type]int
}

// NewABIConfig returns a new ABI configuration for an architecture with
// iRegsCount integer/pointer registers and fRegsCount floating point registers.
func NewABIConfig(iRegsCount, fRegsCount int, offsetForLocals int64) *ABIConfig {
	return &ABIConfig{offsetForLocals: offsetForLocals, regAmounts: RegAmounts{iRegsCount, fRegsCount}, regsForTypeCache: make(map[*types.Type]int)}
}

// Copy returns a copy of an ABIConfig for use in a function's compilation so that access to the cache does not need to be protected with a mutex.
func (a *ABIConfig) Copy() *ABIConfig {
	b := *a
	b.regsForTypeCache = make(map[*types.Type]int)
	return &b
}

// LocalsOffset returns the architecture-dependent offset from SP for args and results.
// In theory this is only used for debugging; it ought to already be incorporated into
// results from the ABI-related methods
func (a *ABIConfig) LocalsOffset() int64 {
	return a.offsetForLocals
}

// FloatIndexFor translates r into an index in the floating point parameter
// registers.  If the result is negative, the input index was actually for the
// integer parameter registers.
func (a *ABIConfig) FloatIndexFor(r RegIndex) int64 {
	return int64(r) - int64(a.regAmounts.intRegs)
}

// NumParamRegs returns the number of parameter registers used for a given type,
// without regard for the number available.
func (a *ABIConfig) NumParamRegs(t *types.Type) int {
	var n int
	if n, ok := a.regsForTypeCache[t]; ok {
		return n
	}

	if t.IsScalar() || t.IsPtrShaped() {
		if t.IsComplex() {
			n = 2
		} else {
			n = (int(t.Size()) + types.RegSize - 1) / types.RegSize
		}
	} else {
		typ := t.Kind()
		switch typ {
		case types.TARRAY:
			n = a.NumParamRegs(t.Elem()) * int(t.NumElem())
		case types.TSTRUCT:
			for _, f := range t.FieldSlice() {
				n += a.NumParamRegs(f.Type)
			}
		case types.TSLICE:
			n = a.NumParamRegs(synthSlice)
		case types.TSTRING:
			n = a.NumParamRegs(synthString)
		case types.TINTER:
			n = a.NumParamRegs(synthIface)
		}
	}
	a.regsForTypeCache[t] = n

	return n
}

// preAllocateParams gets the slice sizes right for inputs and outputs.
func (a *ABIParamResultInfo) preAllocateParams(hasRcvr bool, nIns, nOuts int) {
	if hasRcvr {
		nIns++
	}
	a.inparams = make([]ABIParamAssignment, 0, nIns)
	a.outparams = make([]ABIParamAssignment, 0, nOuts)
}

// ABIAnalyzeTypes takes an optional receiver type, arrays of ins and outs, and returns an ABIParamResultInfo,
// based on the given configuration.  This is the same result computed by config.ABIAnalyze applied to the
// corresponding method/function type, except that all the embedded parameter names are nil.
// This is intended for use by ssagen/ssa.go:(*state).rtcall, for runtime functions that lack a parsed function type.
func (config *ABIConfig) ABIAnalyzeTypes(rcvr *types.Type, ins, outs []*types.Type) *ABIParamResultInfo {
	setup()
	s := assignState{
		stackOffset: config.offsetForLocals,
		rTotal:      config.regAmounts,
	}
	result := &ABIParamResultInfo{config: config}
	result.preAllocateParams(rcvr != nil, len(ins), len(outs))

	// Receiver
	if rcvr != nil {
		result.inparams = append(result.inparams,
			s.assignParamOrReturn(rcvr, nil, false))
	}

	// Inputs
	for _, t := range ins {
		result.inparams = append(result.inparams,
			s.assignParamOrReturn(t, nil, false))
	}
	s.stackOffset = types.Rnd(s.stackOffset, int64(types.RegSize))
	result.inRegistersUsed = s.rUsed.intRegs + s.rUsed.floatRegs

	// Outputs
	s.rUsed = RegAmounts{}
	for _, t := range outs {
		result.outparams = append(result.outparams, s.assignParamOrReturn(t, nil, true))
	}
	// The spill area is at a register-aligned offset and its size is rounded up to a register alignment.
	// TODO in theory could align offset only to minimum required by spilled data types.
	result.offsetToSpillArea = alignTo(s.stackOffset, types.RegSize)
	result.spillAreaSize = alignTo(s.spillOffset, types.RegSize)
	result.outRegistersUsed = s.rUsed.intRegs + s.rUsed.floatRegs

	return result
}

// ABIAnalyzeFuncType takes a function type 'ft' and an ABI rules description
// 'config' and analyzes the function to determine how its parameters
// and results will be passed (in registers or on the stack), returning
// an ABIParamResultInfo object that holds the results of the analysis.
func (config *ABIConfig) ABIAnalyzeFuncType(ft *types.Func) *ABIParamResultInfo {
	setup()
	s := assignState{
		stackOffset: config.offsetForLocals,
		rTotal:      config.regAmounts,
	}
	result := &ABIParamResultInfo{config: config}
	result.preAllocateParams(ft.Receiver != nil, ft.Params.NumFields(), ft.Results.NumFields())

	// Receiver
	// TODO(register args) ? seems like "struct" and "fields" is not right anymore for describing function parameters
	if ft.Receiver != nil && ft.Receiver.NumFields() != 0 {
		r := ft.Receiver.FieldSlice()[0]
		result.inparams = append(result.inparams,
			s.assignParamOrReturn(r.Type, r.Nname, false))
	}

	// Inputs
	ifsl := ft.Params.FieldSlice()
	for _, f := range ifsl {
		result.inparams = append(result.inparams,
			s.assignParamOrReturn(f.Type, f.Nname, false))
	}
	s.stackOffset = types.Rnd(s.stackOffset, int64(types.RegSize))
	result.inRegistersUsed = s.rUsed.intRegs + s.rUsed.floatRegs

	// Outputs
	s.rUsed = RegAmounts{}
	ofsl := ft.Results.FieldSlice()
	for _, f := range ofsl {
		result.outparams = append(result.outparams, s.assignParamOrReturn(f.Type, f.Nname, true))
	}
	// The spill area is at a register-aligned offset and its size is rounded up to a register alignment.
	// TODO in theory could align offset only to minimum required by spilled data types.
	result.offsetToSpillArea = alignTo(s.stackOffset, types.RegSize)
	result.spillAreaSize = alignTo(s.spillOffset, types.RegSize)
	result.outRegistersUsed = s.rUsed.intRegs + s.rUsed.floatRegs
	return result
}

// ABIAnalyze returns the same result as ABIAnalyzeFuncType, but also
// updates the offsets of all the receiver, input, and output fields.
func (config *ABIConfig) ABIAnalyze(t *types.Type) *ABIParamResultInfo {
	ft := t.FuncType()
	result := config.ABIAnalyzeFuncType(ft)
	// Fill in the frame offsets for receiver, inputs, results
	k := 0
	if t.NumRecvs() != 0 {
		config.updateOffset(result, ft.Receiver.FieldSlice()[0], result.inparams[0], false)
		k++
	}
	for i, f := range ft.Params.FieldSlice() {
		config.updateOffset(result, f, result.inparams[k+i], false)
	}
	for i, f := range ft.Results.FieldSlice() {
		config.updateOffset(result, f, result.outparams[i], true)
	}
	return result
}

// parameterUpdateMu protects the Offset field of function/method parameters (a subset of structure Fields)
var parameterUpdateMu sync.Mutex

// FieldOffsetOf returns a concurency-safe version of f.Offset
func FieldOffsetOf(f *types.Field) int64 {
	parameterUpdateMu.Lock()
	defer parameterUpdateMu.Unlock()
	return f.Offset
}

func (config *ABIConfig) updateOffset(result *ABIParamResultInfo, f *types.Field, a ABIParamAssignment, isReturn bool) {
	// Everything except return values in registers has either a frame home (if not in a register) or a frame spill location.
	if !isReturn || len(a.Registers) == 0 {
		// The type frame offset DOES NOT show effects of minimum frame size.
		// Getting this wrong breaks stackmaps, see liveness/plive.go:WriteFuncMap and typebits/typebits.go:Set
		parameterUpdateMu.Lock()
		defer parameterUpdateMu.Unlock()
		off := a.FrameOffset(result) - config.LocalsOffset()
		fOffset := f.Offset
		if fOffset == types.BOGUS_FUNARG_OFFSET {
			// Set the Offset the first time. After that, we may recompute it, but it should never change.
			f.Offset = off
			if f.Nname != nil {
				f.Nname.(*ir.Name).SetFrameOffset(off)
			}
		} else if fOffset != off {
			panic(fmt.Errorf("Offset changed from %d to %d", fOffset, off))
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

// toString method renders an ABIParamAssignment in human-readable
// form, suitable for debugging or unit testing.
func (ri *ABIParamAssignment) toString(config *ABIConfig) string {
	regs := "R{"
	offname := "spilloffset" // offset is for spill for register(s)
	if len(ri.Registers) == 0 {
		offname = "offset" // offset is for memory arg
	}
	for _, r := range ri.Registers {
		regs += " " + config.regAmounts.regString(r)
	}
	return fmt.Sprintf("%s } %s: %d typ: %v", regs, offname, ri.offset, ri.Type)
}

// toString method renders an ABIParamResultInfo in human-readable
// form, suitable for debugging or unit testing.
func (ri *ABIParamResultInfo) String() string {
	res := ""
	for k, p := range ri.inparams {
		res += fmt.Sprintf("IN %d: %s\n", k, p.toString(ri.config))
	}
	for k, r := range ri.outparams {
		res += fmt.Sprintf("OUT %d: %s\n", k, r.toString(ri.config))
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
	pUsed       RegAmounts // regs used by the current param (or pieces therein)
	stackOffset int64      // current stack offset
	spillOffset int64      // current spill offset
}

// align returns a rounded up to t's alignment
func align(a int64, t *types.Type) int64 {
	return alignTo(a, int(t.Align))
}

// alignTo returns a rounded up to t, where t must be 0 or a power of 2.
func alignTo(a int64, t int) int64 {
	if t == 0 {
		return a
	}
	return types.Rnd(a, int64(t))
}

// stackSlot returns a stack offset for a param or result of the
// specified type.
func (state *assignState) stackSlot(t *types.Type) int64 {
	rv := align(state.stackOffset, t)
	state.stackOffset = rv + t.Width
	return rv
}

// allocateRegs returns a set of register indices for a parameter or result
// that we've just determined to be register-assignable. The number of registers
// needed is assumed to be stored in state.pUsed.
func (state *assignState) allocateRegs() []RegIndex {
	regs := []RegIndex{}

	// integer
	for r := state.rUsed.intRegs; r < state.rUsed.intRegs+state.pUsed.intRegs; r++ {
		regs = append(regs, RegIndex(r))
	}
	state.rUsed.intRegs += state.pUsed.intRegs

	// floating
	for r := state.rUsed.floatRegs; r < state.rUsed.floatRegs+state.pUsed.floatRegs; r++ {
		regs = append(regs, RegIndex(r+state.rTotal.intRegs))
	}
	state.rUsed.floatRegs += state.pUsed.floatRegs

	return regs
}

// regAllocate creates a register ABIParamAssignment object for a param
// or result with the specified type, as a final step (this assumes
// that all of the safety/suitability analysis is complete).
func (state *assignState) regAllocate(t *types.Type, name types.Object, isReturn bool) ABIParamAssignment {
	spillLoc := int64(-1)
	if !isReturn {
		// Spill for register-resident t must be aligned for storage of a t.
		spillLoc = align(state.spillOffset, t)
		state.spillOffset = spillLoc + t.Size()
	}
	return ABIParamAssignment{
		Type:      t,
		Name:      name,
		Registers: state.allocateRegs(),
		offset:    int32(spillLoc),
	}
}

// stackAllocate creates a stack memory ABIParamAssignment object for
// a param or result with the specified type, as a final step (this
// assumes that all of the safety/suitability analysis is complete).
func (state *assignState) stackAllocate(t *types.Type, name types.Object) ABIParamAssignment {
	return ABIParamAssignment{
		Type:   t,
		Name:   name,
		offset: int32(state.stackSlot(t)),
	}
}

// intUsed returns the number of integer registers consumed
// at a given point within an assignment stage.
func (state *assignState) intUsed() int {
	return state.rUsed.intRegs + state.pUsed.intRegs
}

// floatUsed returns the number of floating point registers consumed at
// a given point within an assignment stage.
func (state *assignState) floatUsed() int {
	return state.rUsed.floatRegs + state.pUsed.floatRegs
}

// regassignIntegral examines a param/result of integral type 't' to
// determines whether it can be register-assigned. Returns TRUE if we
// can register allocate, FALSE otherwise (and updates state
// accordingly).
func (state *assignState) regassignIntegral(t *types.Type) bool {
	regsNeeded := int(types.Rnd(t.Width, int64(types.PtrSize)) / int64(types.PtrSize))
	if t.IsComplex() {
		regsNeeded = 2
	}

	// Floating point and complex.
	if t.IsFloat() || t.IsComplex() {
		if regsNeeded+state.floatUsed() > state.rTotal.floatRegs {
			// not enough regs
			return false
		}
		state.pUsed.floatRegs += regsNeeded
		return true
	}

	// Non-floating point
	if regsNeeded+state.intUsed() > state.rTotal.intRegs {
		// not enough regs
		return false
	}
	state.pUsed.intRegs += regsNeeded
	return true
}

// regassignArray processes an array type (or array component within some
// other enclosing type) to determine if it can be register assigned.
// Returns TRUE if we can register allocate, FALSE otherwise.
func (state *assignState) regassignArray(t *types.Type) bool {

	nel := t.NumElem()
	if nel == 0 {
		return true
	}
	if nel > 1 {
		// Not an array of length 1: stack assign
		return false
	}
	// Visit element
	return state.regassign(t.Elem())
}

// regassignStruct processes a struct type (or struct component within
// some other enclosing type) to determine if it can be register
// assigned. Returns TRUE if we can register allocate, FALSE otherwise.
func (state *assignState) regassignStruct(t *types.Type) bool {
	for _, field := range t.FieldSlice() {
		if !state.regassign(field.Type) {
			return false
		}
	}
	return true
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
		unsp := types.Types[types.TUNSAFEPTR]
		ui := types.Types[types.TUINTPTR]
		synthSlice = types.NewStruct(types.NoPkg, []*types.Field{
			types.NewField(nxp, fname("ptr"), unsp),
			types.NewField(nxp, fname("len"), ui),
			types.NewField(nxp, fname("cap"), ui),
		})
		synthString = types.NewStruct(types.NoPkg, []*types.Field{
			types.NewField(nxp, fname("data"), unsp),
			types.NewField(nxp, fname("len"), ui),
		})
		synthIface = types.NewStruct(types.NoPkg, []*types.Field{
			types.NewField(nxp, fname("f1"), unsp),
			types.NewField(nxp, fname("f2"), unsp),
		})
	})
}

// regassign examines a given param type (or component within some
// composite) to determine if it can be register assigned.  Returns
// TRUE if we can register allocate, FALSE otherwise.
func (state *assignState) regassign(pt *types.Type) bool {
	typ := pt.Kind()
	if pt.IsScalar() || pt.IsPtrShaped() {
		return state.regassignIntegral(pt)
	}
	switch typ {
	case types.TARRAY:
		return state.regassignArray(pt)
	case types.TSTRUCT:
		return state.regassignStruct(pt)
	case types.TSLICE:
		return state.regassignStruct(synthSlice)
	case types.TSTRING:
		return state.regassignStruct(synthString)
	case types.TINTER:
		return state.regassignStruct(synthIface)
	default:
		panic("not expected")
	}
}

// assignParamOrReturn processes a given receiver, param, or result
// of field f to determine whether it can be register assigned.
// The result of the analysis is recorded in the result
// ABIParamResultInfo held in 'state'.
func (state *assignState) assignParamOrReturn(pt *types.Type, n types.Object, isReturn bool) ABIParamAssignment {
	state.pUsed = RegAmounts{}
	if pt.Width == types.BADWIDTH {
		panic("should never happen")
	} else if pt.Width == 0 {
		return state.stackAllocate(pt, n)
	} else if state.regassign(pt) {
		return state.regAllocate(pt, n, isReturn)
	} else {
		return state.stackAllocate(pt, n)
	}
}
