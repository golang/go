// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ogle

import (
	"debug/elf"
	"debug/gosym"
	"debug/proc"
	"exp/eval"
	"fmt"
	"log"
	"os"
	"reflect"
)

// A FormatError indicates a failure to process information in or
// about a remote process, such as unexpected or missing information
// in the object file or runtime structures.
type FormatError string

func (e FormatError) String() string { return string(e) }

// An UnknownArchitecture occurs when trying to load an object file
// that indicates an architecture not supported by the debugger.
type UnknownArchitecture elf.Machine

func (e UnknownArchitecture) String() string {
	return "unknown architecture: " + elf.Machine(e).String()
}

// A ProcessNotStopped error occurs when attempting to read or write
// memory or registers of a process that is not stopped.
type ProcessNotStopped struct{}

func (e ProcessNotStopped) String() string { return "process not stopped" }

// An UnknownGoroutine error is an internal error representing an
// unrecognized G structure pointer.
type UnknownGoroutine struct {
	OSThread  proc.Thread
	Goroutine proc.Word
}

func (e UnknownGoroutine) String() string {
	return fmt.Sprintf("internal error: unknown goroutine (G %#x)", e.Goroutine)
}

// A NoCurrentGoroutine error occurs when no goroutine is currently
// selected in a process (or when there are no goroutines in a
// process).
type NoCurrentGoroutine struct{}

func (e NoCurrentGoroutine) String() string { return "no current goroutine" }

// A Process represents a remote attached process.
type Process struct {
	Arch
	proc proc.Process

	// The symbol table of this process
	syms *gosym.Table

	// A possibly-stopped OS thread, or nil
	threadCache proc.Thread

	// Types parsed from the remote process
	types map[proc.Word]*remoteType

	// Types and values from the remote runtime package
	runtime runtimeValues

	// Runtime field indexes
	f runtimeIndexes

	// Globals from the sys package (or from no package)
	sys struct {
		lessstack, goexit, newproc, deferproc, newprocreadylocked *gosym.Func
		allg                                                      remotePtr
		g0                                                        remoteStruct
	}

	// Event queue
	posted  []Event
	pending []Event
	event   Event

	// Event hooks
	breakpointHooks     map[proc.Word]*breakpointHook
	goroutineCreateHook *goroutineCreateHook
	goroutineExitHook   *goroutineExitHook

	// Current goroutine, or nil if there are no goroutines
	curGoroutine *Goroutine

	// Goroutines by the address of their G structure
	goroutines map[proc.Word]*Goroutine
}

/*
 * Process creation
 */

// NewProcess constructs a new remote process around a traced
// process, an architecture, and a symbol table.
func NewProcess(tproc proc.Process, arch Arch, syms *gosym.Table) (*Process, os.Error) {
	p := &Process{
		Arch:                arch,
		proc:                tproc,
		syms:                syms,
		types:               make(map[proc.Word]*remoteType),
		breakpointHooks:     make(map[proc.Word]*breakpointHook),
		goroutineCreateHook: new(goroutineCreateHook),
		goroutineExitHook:   new(goroutineExitHook),
		goroutines:          make(map[proc.Word]*Goroutine),
	}

	// Fill in remote runtime
	p.bootstrap()

	switch {
	case p.sys.allg.addr().base == 0:
		return nil, FormatError("failed to find runtime symbol 'allg'")
	case p.sys.g0.addr().base == 0:
		return nil, FormatError("failed to find runtime symbol 'g0'")
	case p.sys.newprocreadylocked == nil:
		return nil, FormatError("failed to find runtime symbol 'newprocreadylocked'")
	case p.sys.goexit == nil:
		return nil, FormatError("failed to find runtime symbol 'sys.goexit'")
	}

	// Get current goroutines
	p.goroutines[p.sys.g0.addr().base] = &Goroutine{p.sys.g0, nil, false}
	err := try(func(a aborter) {
		g := p.sys.allg.aGet(a)
		for g != nil {
			gs := g.(remoteStruct)
			fmt.Printf("*** Found goroutine at %#x\n", gs.addr().base)
			p.goroutines[gs.addr().base] = &Goroutine{gs, nil, false}
			g = gs.field(p.f.G.Alllink).(remotePtr).aGet(a)
		}
	})
	if err != nil {
		return nil, err
	}

	// Create internal breakpoints to catch new and exited goroutines
	p.OnBreakpoint(proc.Word(p.sys.newprocreadylocked.Entry)).(*breakpointHook).addHandler(readylockedBP, true)
	p.OnBreakpoint(proc.Word(p.sys.goexit.Entry)).(*breakpointHook).addHandler(goexitBP, true)

	// Select current frames
	for _, g := range p.goroutines {
		g.resetFrame()
	}

	p.selectSomeGoroutine()

	return p, nil
}

func elfGoSyms(f *elf.File) (*gosym.Table, os.Error) {
	text := f.Section(".text")
	symtab := f.Section(".gosymtab")
	pclntab := f.Section(".gopclntab")
	if text == nil || symtab == nil || pclntab == nil {
		return nil, nil
	}

	symdat, err := symtab.Data()
	if err != nil {
		return nil, err
	}
	pclndat, err := pclntab.Data()
	if err != nil {
		return nil, err
	}

	pcln := gosym.NewLineTable(pclndat, text.Addr)
	tab, err := gosym.NewTable(symdat, pcln)
	if err != nil {
		return nil, err
	}

	return tab, nil
}

// NewProcessElf constructs a new remote process around a traced
// process and the process' ELF object.
func NewProcessElf(tproc proc.Process, f *elf.File) (*Process, os.Error) {
	syms, err := elfGoSyms(f)
	if err != nil {
		return nil, err
	}
	if syms == nil {
		return nil, FormatError("Failed to find symbol table")
	}
	var arch Arch
	switch f.Machine {
	case elf.EM_X86_64:
		arch = Amd64
	default:
		return nil, UnknownArchitecture(f.Machine)
	}
	return NewProcess(tproc, arch, syms)
}

// bootstrap constructs the runtime structure of a remote process.
func (p *Process) bootstrap() {
	// Manually construct runtime types
	p.runtime.String = newManualType(eval.TypeOfNative(rt1String{}), p.Arch)
	p.runtime.Slice = newManualType(eval.TypeOfNative(rt1Slice{}), p.Arch)
	p.runtime.Eface = newManualType(eval.TypeOfNative(rt1Eface{}), p.Arch)

	p.runtime.Type = newManualType(eval.TypeOfNative(rt1Type{}), p.Arch)
	p.runtime.CommonType = newManualType(eval.TypeOfNative(rt1CommonType{}), p.Arch)
	p.runtime.UncommonType = newManualType(eval.TypeOfNative(rt1UncommonType{}), p.Arch)
	p.runtime.StructField = newManualType(eval.TypeOfNative(rt1StructField{}), p.Arch)
	p.runtime.StructType = newManualType(eval.TypeOfNative(rt1StructType{}), p.Arch)
	p.runtime.PtrType = newManualType(eval.TypeOfNative(rt1PtrType{}), p.Arch)
	p.runtime.ArrayType = newManualType(eval.TypeOfNative(rt1ArrayType{}), p.Arch)
	p.runtime.SliceType = newManualType(eval.TypeOfNative(rt1SliceType{}), p.Arch)

	p.runtime.Stktop = newManualType(eval.TypeOfNative(rt1Stktop{}), p.Arch)
	p.runtime.Gobuf = newManualType(eval.TypeOfNative(rt1Gobuf{}), p.Arch)
	p.runtime.G = newManualType(eval.TypeOfNative(rt1G{}), p.Arch)

	// Get addresses of type.*runtime.XType for discrimination.
	rtv := reflect.Indirect(reflect.NewValue(&p.runtime)).(*reflect.StructValue)
	rtvt := rtv.Type().(*reflect.StructType)
	for i := 0; i < rtv.NumField(); i++ {
		n := rtvt.Field(i).Name
		if n[0] != 'P' || n[1] < 'A' || n[1] > 'Z' {
			continue
		}
		sym := p.syms.LookupSym("type.*runtime." + n[1:])
		if sym == nil {
			continue
		}
		rtv.Field(i).(*reflect.UintValue).Set(sym.Value)
	}

	// Get runtime field indexes
	fillRuntimeIndexes(&p.runtime, &p.f)

	// Fill G status
	p.runtime.runtimeGStatus = rt1GStatus

	// Get globals
	p.sys.lessstack = p.syms.LookupFunc("sys.lessstack")
	p.sys.goexit = p.syms.LookupFunc("goexit")
	p.sys.newproc = p.syms.LookupFunc("sys.newproc")
	p.sys.deferproc = p.syms.LookupFunc("sys.deferproc")
	p.sys.newprocreadylocked = p.syms.LookupFunc("newprocreadylocked")
	if allg := p.syms.LookupSym("allg"); allg != nil {
		p.sys.allg = remotePtr{remote{proc.Word(allg.Value), p}, p.runtime.G}
	}
	if g0 := p.syms.LookupSym("g0"); g0 != nil {
		p.sys.g0 = p.runtime.G.mk(remote{proc.Word(g0.Value), p}).(remoteStruct)
	}
}

func (p *Process) selectSomeGoroutine() {
	// Once we have friendly goroutine ID's, there might be a more
	// reasonable behavior for this.
	p.curGoroutine = nil
	for _, g := range p.goroutines {
		if !g.isG0() && g.frame != nil {
			p.curGoroutine = g
			return
		}
	}
}

/*
 * Process memory
 */

func (p *Process) someStoppedOSThread() proc.Thread {
	if p.threadCache != nil {
		if _, err := p.threadCache.Stopped(); err == nil {
			return p.threadCache
		}
	}

	for _, t := range p.proc.Threads() {
		if _, err := t.Stopped(); err == nil {
			p.threadCache = t
			return t
		}
	}
	return nil
}

func (p *Process) Peek(addr proc.Word, out []byte) (int, os.Error) {
	thr := p.someStoppedOSThread()
	if thr == nil {
		return 0, ProcessNotStopped{}
	}
	return thr.Peek(addr, out)
}

func (p *Process) Poke(addr proc.Word, b []byte) (int, os.Error) {
	thr := p.someStoppedOSThread()
	if thr == nil {
		return 0, ProcessNotStopped{}
	}
	return thr.Poke(addr, b)
}

func (p *Process) peekUintptr(a aborter, addr proc.Word) proc.Word {
	return proc.Word(mkUintptr(remote{addr, p}).(remoteUint).aGet(a))
}

/*
 * Events
 */

// OnBreakpoint returns the hook that is run when the program reaches
// the given program counter.
func (p *Process) OnBreakpoint(pc proc.Word) EventHook {
	if bp, ok := p.breakpointHooks[pc]; ok {
		return bp
	}
	// The breakpoint will register itself when a handler is added
	return &breakpointHook{commonHook{nil, 0}, p, pc}
}

// OnGoroutineCreate returns the hook that is run when a goroutine is created.
func (p *Process) OnGoroutineCreate() EventHook {
	return p.goroutineCreateHook
}

// OnGoroutineExit returns the hook that is run when a goroutine exits.
func (p *Process) OnGoroutineExit() EventHook { return p.goroutineExitHook }

// osThreadToGoroutine looks up the goroutine running on an OS thread.
func (p *Process) osThreadToGoroutine(t proc.Thread) (*Goroutine, os.Error) {
	regs, err := t.Regs()
	if err != nil {
		return nil, err
	}
	g := p.G(regs)
	gt, ok := p.goroutines[g]
	if !ok {
		return nil, UnknownGoroutine{t, g}
	}
	return gt, nil
}

// causesToEvents translates the stop causes of the underlying process
// into an event queue.
func (p *Process) causesToEvents() ([]Event, os.Error) {
	// Count causes we're interested in
	nev := 0
	for _, t := range p.proc.Threads() {
		if c, err := t.Stopped(); err == nil {
			switch c := c.(type) {
			case proc.Breakpoint:
				nev++
			case proc.Signal:
				// TODO(austin)
				//nev++;
			}
		}
	}

	// Translate causes to events
	events := make([]Event, nev)
	i := 0
	for _, t := range p.proc.Threads() {
		if c, err := t.Stopped(); err == nil {
			switch c := c.(type) {
			case proc.Breakpoint:
				gt, err := p.osThreadToGoroutine(t)
				if err != nil {
					return nil, err
				}
				events[i] = &Breakpoint{commonEvent{p, gt}, t, proc.Word(c)}
				i++
			case proc.Signal:
				// TODO(austin)
			}
		}
	}

	return events, nil
}

// postEvent appends an event to the posted queue.  These events will
// be processed before any currently pending events.
func (p *Process) postEvent(ev Event) {
	p.posted = append(p.posted, ev)
}

// processEvents processes events in the event queue until no events
// remain, a handler returns EAStop, or a handler returns an error.
// It returns either EAStop or EAContinue and possibly an error.
func (p *Process) processEvents() (EventAction, os.Error) {
	var ev Event
	for len(p.posted) > 0 {
		ev, p.posted = p.posted[0], p.posted[1:]
		action, err := p.processEvent(ev)
		if action == EAStop {
			return action, err
		}
	}

	for len(p.pending) > 0 {
		ev, p.pending = p.pending[0], p.pending[1:]
		action, err := p.processEvent(ev)
		if action == EAStop {
			return action, err
		}
	}

	return EAContinue, nil
}

// processEvent processes a single event, without manipulating the
// event queues.  It returns either EAStop or EAContinue and possibly
// an error.
func (p *Process) processEvent(ev Event) (EventAction, os.Error) {
	p.event = ev

	var action EventAction
	var err os.Error
	switch ev := p.event.(type) {
	case *Breakpoint:
		hook, ok := p.breakpointHooks[ev.pc]
		if !ok {
			break
		}
		p.curGoroutine = ev.Goroutine()
		action, err = hook.handle(ev)

	case *GoroutineCreate:
		p.curGoroutine = ev.Goroutine()
		action, err = p.goroutineCreateHook.handle(ev)

	case *GoroutineExit:
		action, err = p.goroutineExitHook.handle(ev)

	default:
		log.Panicf("Unknown event type %T in queue", p.event)
	}

	if err != nil {
		return EAStop, err
	} else if action == EAStop {
		return EAStop, nil
	}
	return EAContinue, nil
}

// Event returns the last event that caused the process to stop.  This
// may return nil if the process has never been stopped by an event.
//
// TODO(austin) Return nil if the user calls p.Stop()?
func (p *Process) Event() Event { return p.event }

/*
 * Process control
 */

// TODO(austin) Cont, WaitStop, and Stop.  Need to figure out how
// event handling works with these.  Originally I did it only in
// WaitStop, but if you Cont and there are pending events, then you
// have to not actually continue and wait until a WaitStop to process
// them, even if the event handlers will tell you to continue.  We
// could handle them in both Cont and WaitStop to avoid this problem,
// but it's still weird if an event happens after the Cont and before
// the WaitStop that the handlers say to continue from.  Or we could
// handle them on a separate thread.  Then obviously you get weird
// asynchronous things, like prints while the user it typing a command,
// but that's not necessarily a bad thing.

// ContWait resumes process execution and waits for an event to occur
// that stops the process.
func (p *Process) ContWait() os.Error {
	for {
		a, err := p.processEvents()
		if err != nil {
			return err
		} else if a == EAStop {
			break
		}
		err = p.proc.Continue()
		if err != nil {
			return err
		}
		err = p.proc.WaitStop()
		if err != nil {
			return err
		}
		for _, g := range p.goroutines {
			g.resetFrame()
		}
		p.pending, err = p.causesToEvents()
		if err != nil {
			return err
		}
	}
	return nil
}

// Out selects the caller frame of the current frame.
func (p *Process) Out() os.Error {
	if p.curGoroutine == nil {
		return NoCurrentGoroutine{}
	}
	return p.curGoroutine.Out()
}

// In selects the frame called by the current frame.
func (p *Process) In() os.Error {
	if p.curGoroutine == nil {
		return NoCurrentGoroutine{}
	}
	return p.curGoroutine.In()
}
