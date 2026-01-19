// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asmgen

import (
	"fmt"
	"math/bits"
	"slices"
)

// Note: Exported fields and methods are expected to be used
// by function generators (like the ones in add.go and so on).
// Unexported fields and methods should not be.

// A Pipe manages the input and output data pipelines for a function's
// memory operations.
//
// The input is one or more equal-length slices of words, so collectively
// it can be viewed as a matrix, in which each slice is a row and each column
// is a set of corresponding words from the different slices.
// The output can be viewed the same way, although it is often just one row.
type Pipe struct {
	f               *Func    // function being generated
	label           string   // prefix for loop labels (default "loop")
	backward        bool     // processing columns in reverse
	started         bool     // Start has been called
	loaded          bool     // LoadPtrs has been called
	inPtr           []RegPtr // input slice pointers
	hints           []Hint   // for each inPtr, a register hint to use for its data
	outPtr          []RegPtr // output slice pointers
	index           Reg      // index register, if in use
	useIndexCounter bool     // index counter requested
	indexCounter    int      // index is also counter (386); 0 no, -1 negative counter, +1 positive counter
	readOff         int      // read offset not yet added to index
	writeOff        int      // write offset not yet added to index
	factors         []int    // unrolling factors
	counts          []Reg    // iterations for each factor
	needWrite       bool     // need a write call during Loop1/LoopN
	maxColumns      int      // maximum columns during unrolled loop
	unrollStart     func()   // emit code at start of unrolled body
	unrollEnd       func()   // emit code end of unrolled body
}

// Pipe creates and returns a new pipe for use in the function f.
func (f *Func) Pipe() *Pipe {
	a := f.Asm
	p := &Pipe{
		f:          f,
		label:      "loop",
		maxColumns: 10000000,
	}
	if m := a.Arch.maxColumns; m != 0 {
		p.maxColumns = m
	}
	return p
}

// SetBackward sets the pipe to process the input and output columns in reverse order.
// This is needed for left shifts, which might otherwise overwrite data they will read later.
func (p *Pipe) SetBackward() {
	if p.loaded {
		p.f.Asm.Fatalf("SetBackward after Start/LoadPtrs")
	}
	p.backward = true
}

// SetUseIndexCounter sets the pipe to use an index counter if possible,
// meaning the loop counter is also used as an index for accessing the slice data.
// This clever trick is slower on modern processors, but it is still necessary on 386.
// On non-386 systems, SetUseIndexCounter is a no-op.
func (p *Pipe) SetUseIndexCounter() {
	if p.f.Asm.Arch.memIndex == nil { // need memIndex (only 386 provides it)
		return
	}
	p.useIndexCounter = true
}

// SetLabel sets the label prefix for the loops emitted by the pipe.
// The default prefix is "loop".
func (p *Pipe) SetLabel(label string) {
	p.label = label
}

// SetMaxColumns sets the maximum number of
// columns processed in a single loop body call.
func (p *Pipe) SetMaxColumns(m int) {
	p.maxColumns = m
}

// SetHint records that the inputs from the named vector
// should be allocated with the given register hint.
//
// If the hint indicates a single register on the target architecture,
// then SetHint calls SetMaxColumns(1), since the hinted register
// can only be used for one value at a time.
func (p *Pipe) SetHint(name string, hint Hint) {
	if hint == HintMemOK && !p.f.Asm.Arch.memOK {
		return
	}
	i := slices.Index(p.f.inputs, name)
	if i < 0 {
		p.f.Asm.Fatalf("unknown input name %s", name)
	}
	if p.f.Asm.hint(hint) != "" {
		p.SetMaxColumns(1)
	}
	for len(p.hints) <= i {
		p.hints = append(p.hints, HintNone)
	}
	p.hints[i] = hint
}

// LoadPtrs loads the slice pointer arguments into registers,
// assuming that the slice length n has already been loaded
// into the register n.
//
// Start will call LoadPtrs if it has not been called already.
// LoadPtrs only needs to be called explicitly when code needs
// to use LoadN before Start, like when the shift.go generators
// read an initial word before the loop.
func (p *Pipe) LoadPtrs(n Reg) {
	a := p.f.Asm
	if p.loaded {
		a.Fatalf("pointers already loaded")
	}

	// Load the actual pointers.
	p.loaded = true
	for _, name := range p.f.inputs {
		p.inPtr = append(p.inPtr, RegPtr(p.f.Arg(name+"_base")))
	}
	for _, name := range p.f.outputs {
		p.outPtr = append(p.outPtr, RegPtr(p.f.Arg(name+"_base")))
	}

	// Decide the memory access strategy for LoadN and StoreN.
	switch {
	case p.backward && p.useIndexCounter:
		// Generator wants an index counter, meaning when the iteration counter
		// is AX, we will access the slice with pointer BX using (BX)(AX*WordBytes).
		// The loop is moving backward through the slice, but the counter
		// is also moving backward, so not much to do.
		a.Comment("run loop backward, using counter as positive index")
		p.indexCounter = +1
		p.index = n

	case !p.backward && p.useIndexCounter:
		// Generator wants an index counter, but the loop is moving forward.
		// To make the counter move in the direction of data access,
		// we negate the counter, counting up from -len(z) to -1.
		// To make the index access the right words, we add len(z)*WordBytes
		// to each of the pointers.
		// See comment below about the garbage collector (non-)implications
		// of pointing beyond the slice bounds.
		a.Comment("use counter as negative index")
		p.indexCounter = -1
		p.index = n
		for _, ptr := range p.inPtr {
			a.AddWords(n, ptr, ptr)
		}
		for _, ptr := range p.outPtr {
			a.AddWords(n, ptr, ptr)
		}
		a.Neg(n, n)

	case p.backward:
		// Generator wants to run the loop backward.
		// We'll decrement the pointers before using them,
		// so position them at the very end of the slices.
		// If we had precise pointer information for assembly,
		// these pointers would cause problems with the garbage collector,
		// since they no longer point into the allocated slice,
		// but the garbage collector ignores unexpected values in assembly stacks,
		// and the actual slice pointers are still in the argument stack slots,
		// so the slices won't be collected early.
		// If we switched to the register ABI, we might have to rethink this.
		// (The same thing happens by the end of forward loops,
		// but it's less important since once the pointers go off the slice
		// in a forward loop, the loop is over and the slice won't be accessed anymore.)
		a.Comment("run loop backward")
		for _, ptr := range p.inPtr {
			a.AddWords(n, ptr, ptr)
		}
		for _, ptr := range p.outPtr {
			a.AddWords(n, ptr, ptr)
		}

	case !p.backward:
		// Nothing to do!
	}
}

// LoadN returns the next n columns of input words as a slice of rows.
// Regs for inputs that have been marked using p.SetMemOK will be direct memory references.
// Regs for other inputs will be newly allocated registers and must be freed.
func (p *Pipe) LoadN(n int) [][]Reg {
	a := p.f.Asm
	regs := make([][]Reg, len(p.inPtr))
	for i, ptr := range p.inPtr {
		regs[i] = make([]Reg, n)
		switch {
		case a.Arch.loadIncN != nil:
			// Load from memory and advance pointers at the same time.
			for j := range regs[i] {
				regs[i][j] = p.f.Asm.Reg()
			}
			if p.backward {
				a.Arch.loadDecN(a, ptr, regs[i])
			} else {
				a.Arch.loadIncN(a, ptr, regs[i])
			}

		default:
			// Load from memory using offsets.
			// We'll advance the pointers or the index counter later.
			for j := range n {
				off := p.readOff + j
				if p.backward {
					off = -(off + 1)
				}
				var mem Reg
				if p.indexCounter != 0 {
					mem = a.Arch.memIndex(a, off*a.Arch.WordBytes, p.index, ptr)
				} else {
					mem = ptr.mem(off * a.Arch.WordBytes)
				}
				h := HintNone
				if i < len(p.hints) {
					h = p.hints[i]
				}
				if h == HintMemOK {
					regs[i][j] = mem
				} else {
					r := p.f.Asm.RegHint(h)
					a.Mov(mem, r)
					regs[i][j] = r
				}
			}
		}
	}
	p.readOff += n
	return regs
}

// StoreN writes regs (a slice of rows) to the next n columns of output, where n = len(regs[0]).
func (p *Pipe) StoreN(regs [][]Reg) {
	p.needWrite = false
	a := p.f.Asm
	if len(regs) != len(p.outPtr) {
		p.f.Asm.Fatalf("wrong number of output rows")
	}
	n := len(regs[0])
	for i, ptr := range p.outPtr {
		switch {
		case a.Arch.storeIncN != nil:
			// Store to memory and advance pointers at the same time.
			if p.backward {
				a.Arch.storeDecN(a, ptr, regs[i])
			} else {
				a.Arch.storeIncN(a, ptr, regs[i])
			}

		default:
			// Store to memory using offsets.
			// We'll advance the pointers or the index counter later.
			for j, r := range regs[i] {
				off := p.writeOff + j
				if p.backward {
					off = -(off + 1)
				}
				var mem Reg
				if p.indexCounter != 0 {
					mem = a.Arch.memIndex(a, off*a.Arch.WordBytes, p.index, ptr)
				} else {
					mem = ptr.mem(off * a.Arch.WordBytes)
				}
				a.Mov(r, mem)
			}
		}
	}
	p.writeOff += n
}

// advancePtrs advances the pointers by step
// or handles bookkeeping for an imminent index advance by step
// that the caller will do.
func (p *Pipe) advancePtrs(step int) {
	a := p.f.Asm
	switch {
	case a.Arch.loadIncN != nil:
		// nothing to do

	default:
		// Adjust read/write offsets for pointer advance (or imminent index advance).
		p.readOff -= step
		p.writeOff -= step

		if p.indexCounter == 0 {
			// Advance pointers.
			if p.backward {
				step = -step
			}
			for _, ptr := range p.inPtr {
				a.Add(a.Imm(step*a.Arch.WordBytes), Reg(ptr), Reg(ptr), KeepCarry)
			}
			for _, ptr := range p.outPtr {
				a.Add(a.Imm(step*a.Arch.WordBytes), Reg(ptr), Reg(ptr), KeepCarry)
			}
		}
	}
}

// DropInput deletes the named input from the pipe,
// usually because it has been exhausted.
// (This is not used yet but will be used in a future generator.)
func (p *Pipe) DropInput(name string) {
	i := slices.Index(p.f.inputs, name)
	if i < 0 {
		p.f.Asm.Fatalf("unknown input %s", name)
	}
	ptr := p.inPtr[i]
	p.f.Asm.Free(Reg(ptr))
	p.inPtr = slices.Delete(p.inPtr, i, i+1)
	p.f.inputs = slices.Delete(p.f.inputs, i, i+1)
	if len(p.hints) > i {
		p.hints = slices.Delete(p.hints, i, i+1)
	}
}

// Start prepares to loop over n columns.
// The factors give a sequence of unrolling factors to use,
// which must be either strictly increasing or strictly decreasing
// and must include 1.
// For example, 4, 1 means to process 4 elements at a time
// and then 1 at a time for the final 0-3; specifying 1,4 instead
// handles 0-3 elements first and then 4 at a time.
// Similarly, 32, 4, 1 means to process 32 at a time,
// then 4 at a time, then 1 at a time.
//
// One benefit of using 1, 4 instead of 4, 1 is that the body
// processing 4 at a time needs more registers, and if it is
// the final body, the register holding the fragment count (0-3)
// has been freed and is available for use.
//
// Start may modify the carry flag.
//
// Start must be followed by a call to Loop1 or LoopN,
// but it is permitted to emit other instructions first,
// for example to set an initial carry flag.
func (p *Pipe) Start(n Reg, factors ...int) {
	a := p.f.Asm
	if p.started {
		a.Fatalf("loop already started")
	}
	if p.useIndexCounter && len(factors) > 1 {
		a.Fatalf("cannot call SetUseIndexCounter and then use Start with factors != [1]; have factors = %v", factors)
	}
	p.started = true
	if !p.loaded {
		if len(factors) == 1 {
			p.SetUseIndexCounter()
		}
		p.LoadPtrs(n)
	}

	// If there were calls to LoadN between LoadPtrs and Start,
	// adjust the loop not to scan those columns, assuming that
	// either the code already called an equivalent StoreN or else
	// that it will do so after the loop.
	if off := p.readOff; off != 0 {
		if p.indexCounter < 0 {
			// Index is negated, so add off instead of subtracting.
			a.Add(a.Imm(off), n, n, SmashCarry)
		} else {
			a.Sub(a.Imm(off), n, n, SmashCarry)
		}
		if p.indexCounter != 0 {
			// n is also the index we are using, so adjust readOff and writeOff
			// to continue to point at the same positions as before we changed n.
			p.readOff -= off
			p.writeOff -= off
		}
	}

	p.Restart(n, factors...)
}

// Restart prepares to loop over an additional n columns,
// beyond a previous loop run by p.Start/p.Loop.
func (p *Pipe) Restart(n Reg, factors ...int) {
	a := p.f.Asm
	if !p.started {
		a.Fatalf("pipe not started")
	}
	p.factors = factors
	p.counts = make([]Reg, len(factors))
	if len(factors) == 0 {
		factors = []int{1}
	}

	// Compute the loop lengths for each unrolled section into separate registers.
	// We compute them all ahead of time in case the computation would smash
	// a carry flag that the loop bodies need preserved.
	if len(factors) > 1 {
		a.Comment("compute unrolled loop lengths")
	}
	switch {
	default:
		a.Fatalf("invalid factors %v", factors)

	case factors[0] == 1:
		// increasing loop factors
		div := 1
		for i, f := range factors[1:] {
			if f <= factors[i] {
				a.Fatalf("non-increasing factors %v", factors)
			}
			if f&(f-1) != 0 {
				a.Fatalf("non-power-of-two factors %v", factors)
			}
			t := p.f.Asm.Reg()
			f /= div
			a.And(a.Imm(f-1), n, t)
			a.Rsh(a.Imm(bits.TrailingZeros(uint(f))), n, n)
			div *= f
			p.counts[i] = t
		}
		p.counts[len(p.counts)-1] = n

	case factors[len(factors)-1] == 1:
		// decreasing loop factors
		for i, f := range factors[:len(factors)-1] {
			if f <= factors[i+1] {
				a.Fatalf("non-decreasing factors %v", factors)
			}
			if f&(f-1) != 0 {
				a.Fatalf("non-power-of-two factors %v", factors)
			}
			t := p.f.Asm.Reg()
			a.Rsh(a.Imm(bits.TrailingZeros(uint(f))), n, t)
			a.And(a.Imm(f-1), n, n)
			p.counts[i] = t
		}
		p.counts[len(p.counts)-1] = n
	}
}

// Done frees all the registers allocated by the pipe.
func (p *Pipe) Done() {
	for _, ptr := range p.inPtr {
		p.f.Asm.Free(Reg(ptr))
	}
	p.inPtr = nil
	for _, ptr := range p.outPtr {
		p.f.Asm.Free(Reg(ptr))
	}
	p.outPtr = nil
	p.index = Reg{}
}

// Loop emits code for the loop, calling block repeatedly to emit code that
// handles a block of N input columns (for arbitrary N = len(in[0]) chosen by p).
// block must call p.StoreN(out) to write N output columns.
// The out slice is a pre-allocated matrix of uninitialized Reg values.
// block is expected to set each entry to the Reg that should be written
// before calling p.StoreN(out).
//
// For example, if the loop is to be unrolled 4x in blocks of 2 columns each,
// the sequence of calls to emit the unrolled loop body is:
//
//	start()  // set by pAtUnrollStart
//	... reads for 2 columns ...
//	block()
//	... writes for 2 columns ...
//	... reads for 2 columns ...
//	block()
//	... writes for 2 columns ...
//	end()  // set by p.AtUnrollEnd
//
// Any registers allocated during block are freed automatically when block returns.
func (p *Pipe) Loop(block func(in, out [][]Reg)) {
	if p.factors == nil {
		p.f.Asm.Fatalf("Pipe.Start not called")
	}
	for i, factor := range p.factors {
		n := p.counts[i]
		p.unroll(n, factor, block)
		if i < len(p.factors)-1 {
			p.f.Asm.Free(n)
		}
	}
	p.factors = nil
}

// AtUnrollStart sets a function to call at the start of an unrolled sequence.
// See [Pipe.Loop] for details.
func (p *Pipe) AtUnrollStart(start func()) {
	p.unrollStart = start
}

// AtUnrollEnd sets a function to call at the end of an unrolled sequence.
// See [Pipe.Loop] for details.
func (p *Pipe) AtUnrollEnd(end func()) {
	p.unrollEnd = end
}

// unroll emits a single unrolled loop for the given factor, iterating n times.
func (p *Pipe) unroll(n Reg, factor int, block func(in, out [][]Reg)) {
	a := p.f.Asm
	label := fmt.Sprintf("%s%d", p.label, factor)

	// Top of loop control flow.
	a.Label(label)
	if a.Arch.loopTop != "" {
		a.Printf("\t"+a.Arch.loopTop+"\n", n, label+"done")
	} else {
		a.JmpZero(n, label+"done")
	}
	a.Label(label + "cont")

	// Unrolled loop body.
	if factor < p.maxColumns {
		a.Comment("unroll %dX", factor)
	} else {
		a.Comment("unroll %dX in batches of %d", factor, p.maxColumns)
	}
	if p.unrollStart != nil {
		p.unrollStart()
	}
	for done := 0; done < factor; {
		batch := min(factor-done, p.maxColumns)
		regs := a.RegsUsed()
		out := make([][]Reg, len(p.outPtr))
		for i := range out {
			out[i] = make([]Reg, batch)
		}
		in := p.LoadN(batch)
		p.needWrite = true
		block(in, out)
		if p.needWrite && len(p.outPtr) > 0 {
			a.Fatalf("missing p.Write1 or p.StoreN")
		}
		a.SetRegsUsed(regs) // free anything block allocated
		done += batch
	}
	if p.unrollEnd != nil {
		p.unrollEnd()
	}
	p.advancePtrs(factor)

	// Bottom of loop control flow.
	switch {
	case p.indexCounter >= 0 && a.Arch.loopBottom != "":
		a.Printf("\t"+a.Arch.loopBottom+"\n", n, label+"cont")

	case p.indexCounter >= 0:
		a.Sub(a.Imm(1), n, n, KeepCarry)
		a.JmpNonZero(n, label+"cont")

	case p.indexCounter < 0 && a.Arch.loopBottomNeg != "":
		a.Printf("\t"+a.Arch.loopBottomNeg+"\n", n, label+"cont")

	case p.indexCounter < 0:
		a.Add(a.Imm(1), n, n, KeepCarry)
	}
	a.Label(label + "done")
}
