// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ogle

import (
	"debug/proc"
	"exp/eval"
	"fmt"
	"os"
)

// A Goroutine represents a goroutine in a remote process.
type Goroutine struct {
	g     remoteStruct
	frame *Frame
	dead  bool
}

func (t *Goroutine) String() string {
	if t.dead {
		return "<dead thread>"
	}
	// TODO(austin) Give threads friendly ID's, possibly including
	// the name of the entry function.
	return fmt.Sprintf("thread %#x", t.g.addr().base)
}

// isG0 returns true if this thread if the internal idle thread
func (t *Goroutine) isG0() bool { return t.g.addr().base == t.g.r.p.sys.g0.addr().base }

func (t *Goroutine) resetFrame() (err os.Error) {
	// TODO(austin) Reuse any live part of the current frame stack
	// so existing references to Frame's keep working.
	t.frame, err = newFrame(t.g)
	return
}

// Out selects the caller frame of the current frame.
func (t *Goroutine) Out() os.Error {
	f, err := t.frame.Outer()
	if f != nil {
		t.frame = f
	}
	return err
}

// In selects the frame called by the current frame.
func (t *Goroutine) In() os.Error {
	f := t.frame.Inner()
	if f != nil {
		t.frame = f
	}
	return nil
}

func readylockedBP(ev Event) (EventAction, os.Error) {
	b := ev.(*Breakpoint)
	p := b.Process()

	// The new g is the only argument to this function, so the
	// stack will have the return address, then the G*.
	regs, err := b.osThread.Regs()
	if err != nil {
		return EAStop, err
	}
	sp := regs.SP()
	addr := sp + proc.Word(p.PtrSize())
	arg := remotePtr{remote{addr, p}, p.runtime.G}
	var gp eval.Value
	err = try(func(a aborter) { gp = arg.aGet(a) })
	if err != nil {
		return EAStop, err
	}
	if gp == nil {
		return EAStop, UnknownGoroutine{b.osThread, 0}
	}
	gs := gp.(remoteStruct)
	g := &Goroutine{gs, nil, false}
	p.goroutines[gs.addr().base] = g

	// Enqueue goroutine creation event
	parent := b.Goroutine()
	if parent.isG0() {
		parent = nil
	}
	p.postEvent(&GoroutineCreate{commonEvent{p, g}, parent})

	// If we don't have any thread selected, select this one
	if p.curGoroutine == nil {
		p.curGoroutine = g
	}

	return EADefault, nil
}

func goexitBP(ev Event) (EventAction, os.Error) {
	b := ev.(*Breakpoint)
	p := b.Process()

	g := b.Goroutine()
	g.dead = true

	addr := g.g.addr().base
	p.goroutines[addr] = nil, false

	// Enqueue thread exit event
	p.postEvent(&GoroutineExit{commonEvent{p, g}})

	// If we just exited our selected goroutine, selected another
	if p.curGoroutine == g {
		p.selectSomeGoroutine()
	}

	return EADefault, nil
}
