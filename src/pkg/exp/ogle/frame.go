// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ogle

import (
	"debug/gosym"
	"debug/proc"
	"fmt"
	"os"
)

// A Frame represents a single frame on a remote call stack.
type Frame struct {
	// pc is the PC of the next instruction that will execute in
	// this frame.  For lower frames, this is the instruction
	// following the CALL instruction.
	pc, sp, fp proc.Word
	// The runtime.Stktop of the active stack segment
	stk remoteStruct
	// The function this stack frame is in
	fn *gosym.Func
	// The path and line of the CALL or current instruction.  Note
	// that this differs slightly from the meaning of Frame.pc.
	path string
	line int
	// The inner and outer frames of this frame.  outer is filled
	// in lazily.
	inner, outer *Frame
}

// newFrame returns the top-most Frame of the given g's thread.
func newFrame(g remoteStruct) (*Frame, os.Error) {
	var f *Frame
	err := try(func(a aborter) { f = aNewFrame(a, g) })
	return f, err
}

func aNewFrame(a aborter, g remoteStruct) *Frame {
	p := g.r.p
	var pc, sp proc.Word

	// Is this G alive?
	switch g.field(p.f.G.Status).(remoteInt).aGet(a) {
	case p.runtime.Gidle, p.runtime.Gmoribund, p.runtime.Gdead:
		return nil
	}

	// Find the OS thread for this G

	// TODO(austin) Ideally, we could look at the G's state and
	// figure out if it's on an OS thread or not.  However, this
	// is difficult because the state isn't updated atomically
	// with scheduling changes.
	for _, t := range p.proc.Threads() {
		regs, err := t.Regs()
		if err != nil {
			// TODO(austin) What to do?
			continue
		}
		thisg := p.G(regs)
		if thisg == g.addr().base {
			// Found this G's OS thread
			pc = regs.PC()
			sp = regs.SP()

			// If this thread crashed, try to recover it
			if pc == 0 {
				pc = p.peekUintptr(a, pc)
				sp += 8
			}

			break
		}
	}

	if pc == 0 && sp == 0 {
		// G is not mapped to an OS thread.  Use the
		// scheduler's stored PC and SP.
		sched := g.field(p.f.G.Sched).(remoteStruct)
		pc = proc.Word(sched.field(p.f.Gobuf.Pc).(remoteUint).aGet(a))
		sp = proc.Word(sched.field(p.f.Gobuf.Sp).(remoteUint).aGet(a))
	}

	// Get Stktop
	stk := g.field(p.f.G.Stackbase).(remotePtr).aGet(a).(remoteStruct)

	return prepareFrame(a, pc, sp, stk, nil)
}

// prepareFrame creates a Frame from the PC and SP within that frame,
// as well as the active stack segment.  This function takes care of
// traversing stack breaks and unwinding closures.
func prepareFrame(a aborter, pc, sp proc.Word, stk remoteStruct, inner *Frame) *Frame {
	// Based on src/pkg/runtime/amd64/traceback.c:traceback
	p := stk.r.p
	top := inner == nil

	// Get function
	var path string
	var line int
	var fn *gosym.Func

	for i := 0; i < 100; i++ {
		// Traverse segmented stack breaks
		if p.sys.lessstack != nil && pc == proc.Word(p.sys.lessstack.Value) {
			// Get stk->gobuf.pc
			pc = proc.Word(stk.field(p.f.Stktop.Gobuf).(remoteStruct).field(p.f.Gobuf.Pc).(remoteUint).aGet(a))
			// Get stk->gobuf.sp
			sp = proc.Word(stk.field(p.f.Stktop.Gobuf).(remoteStruct).field(p.f.Gobuf.Sp).(remoteUint).aGet(a))
			// Get stk->stackbase
			stk = stk.field(p.f.Stktop.Stackbase).(remotePtr).aGet(a).(remoteStruct)
			continue
		}

		// Get the PC of the call instruction
		callpc := pc
		if !top && (p.sys.goexit == nil || pc != proc.Word(p.sys.goexit.Value)) {
			callpc--
		}

		// Look up function
		path, line, fn = p.syms.PCToLine(uint64(callpc))
		if fn != nil {
			break
		}

		// Closure?
		var buf = make([]byte, p.ClosureSize())
		if _, err := p.Peek(pc, buf); err != nil {
			break
		}
		spdelta, ok := p.ParseClosure(buf)
		if ok {
			sp += proc.Word(spdelta)
			pc = p.peekUintptr(a, sp-proc.Word(p.PtrSize()))
		}
	}
	if fn == nil {
		return nil
	}

	// Compute frame pointer
	var fp proc.Word
	if fn.FrameSize < p.PtrSize() {
		fp = sp + proc.Word(p.PtrSize())
	} else {
		fp = sp + proc.Word(fn.FrameSize)
	}
	// TODO(austin) To really figure out if we're in the prologue,
	// we need to disassemble the function and look for the call
	// to morestack.  For now, just special case the entry point.
	//
	// TODO(austin) What if we're in the call to morestack in the
	// prologue?  Then top == false.
	if top && pc == proc.Word(fn.Entry) {
		// We're in the function prologue, before SP
		// has been adjusted for the frame.
		fp -= proc.Word(fn.FrameSize - p.PtrSize())
	}

	return &Frame{pc, sp, fp, stk, fn, path, line, inner, nil}
}

// Outer returns the Frame that called this Frame, or nil if this is
// the outermost frame.
func (f *Frame) Outer() (*Frame, os.Error) {
	var fr *Frame
	err := try(func(a aborter) { fr = f.aOuter(a) })
	return fr, err
}

func (f *Frame) aOuter(a aborter) *Frame {
	// Is there a cached outer frame
	if f.outer != nil {
		return f.outer
	}

	p := f.stk.r.p

	sp := f.fp
	if f.fn == p.sys.newproc && f.fn == p.sys.deferproc {
		// TODO(rsc) The compiler inserts two push/pop's
		// around calls to go and defer.  Russ says this
		// should get fixed in the compiler, but we account
		// for it for now.
		sp += proc.Word(2 * p.PtrSize())
	}

	pc := p.peekUintptr(a, f.fp-proc.Word(p.PtrSize()))
	if pc < 0x1000 {
		return nil
	}

	// TODO(austin) Register this frame for shoot-down.

	f.outer = prepareFrame(a, pc, sp, f.stk, f)
	return f.outer
}

// Inner returns the Frame called by this Frame, or nil if this is the
// innermost frame.
func (f *Frame) Inner() *Frame { return f.inner }

func (f *Frame) String() string {
	res := f.fn.Name
	if f.pc > proc.Word(f.fn.Value) {
		res += fmt.Sprintf("+%#x", f.pc-proc.Word(f.fn.Entry))
	}
	return res + fmt.Sprintf(" %s:%d", f.path, f.line)
}
