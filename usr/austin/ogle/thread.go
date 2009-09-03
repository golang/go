// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ogle

import (
	"fmt";
	"os";
	"ptrace";
)

// A Thread represents a Go thread.
type Thread struct {
	g remoteStruct;
	frame *Frame;
	dead bool;
}

func (t *Thread) String() string {
	if t.dead {
		return "<dead thread>";
	}
	// TODO(austin) Give threads friendly ID's
	return fmt.Sprintf("thread %#x", t.g.addr().base);
}

// isG0 returns true if this thread if the internal idle thread
func (t *Thread) isG0() bool {
	return t.g.addr().base == t.g.r.p.sys.g0.addr().base;
}

func (t *Thread) resetFrame() {
	// TODO(austin) NewFrame can abort
	// TODO(austin) Reuse any live part of the current frame stack
	// so existing references to Frame's keep working.
	t.frame = NewFrame(t.g);
}

// Out selects the caller frame of the current frame.
func (t *Thread) Out() os.Error {
	// TODO(austin) Outer can abort
	f := t.frame.Outer();
	if f != nil {
		t.frame = f;
	}
	return nil;
}

// In selects the frame called by the current frame.
func (t *Thread) In() os.Error {
	f := t.frame.Inner();
	if f != nil {
		t.frame = f;
	}
	return nil;
}

func readylockedBP(ev Event) (EventAction, os.Error) {
	b := ev.(*Breakpoint);
	p := b.Process();

	// The new g is the only argument to this function, so the
	// stack will have the return address, then the G*.
	regs, err := b.osThread.Regs();
	if err != nil {
		return EAStop, err;
	}
	sp := regs.SP();
	addr := sp + ptrace.Word(p.PtrSize());
	arg := remotePtr{remote{addr, p}, p.runtime.G};
	g := arg.Get();
	if g == nil {
		return EAStop, UnknownThread{b.osThread, 0};
	}
	gs := g.(remoteStruct);
	t := &Thread{gs, nil, false};
	p.threads[gs.addr().base] = t;

	// Enqueue thread creation event
	parent := b.Thread();
	if parent.isG0() {
		parent = nil;
	}
	p.postEvent(&ThreadCreate{commonEvent{p, t}, parent});

	// If we don't have any thread selected, select this one
	if p.curThread == nil {
		p.curThread = t;
	}

	return EADefault, nil;
}

func goexitBP(ev Event) (EventAction, os.Error) {
	b := ev.(*Breakpoint);
	p := b.Process();

	t := b.Thread();
	t.dead = true;

	addr := t.g.addr().base;
	p.threads[addr] = nil, false;

	// Enqueue thread exit event
	p.postEvent(&ThreadExit{commonEvent{p, t}});

	// If we just exited our selected thread, selected another
	if p.curThread == t {
		p.selectSomeThread();
	}

	return EADefault, nil;
}
