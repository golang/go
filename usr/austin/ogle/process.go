// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ogle

import (
	"eval";
	"ptrace";
	"reflect";
	"os";
	"sym";
)

// A FormatError indicates a failure to process information in or
// about a remote process, such as unexpected or missing information
// in the object file or runtime structures.
type FormatError string

func (e FormatError) String() string {
	return string(e);
}

// An UnknownArchitecture occurs when trying to load an object file
// that indicates an architecture not supported by the debugger.
type UnknownArchitecture sym.ElfMachine

func (e UnknownArchitecture) String() string {
	return "unknown architecture: " + sym.ElfMachine(e).String();
}

// A ProcessNotStopped error occurs when attempting to read or write
// memory or registers of a process that is not stopped.
type ProcessNotStopped struct {}

func (e ProcessNotStopped) String() string {
	return "process not stopped";
}

// A Process represents a remote attached process.
type Process struct {
	Arch;
	ptrace.Process;

	// The symbol table of this process
	syms *sym.GoSymTable;

	// Current frame, or nil if the current thread is not stopped
	frame *Frame;

	// A possibly-stopped OS thread, or nil
	threadCache ptrace.Thread;

	// Types parsed from the remote process
	types map[ptrace.Word] *remoteType;

	// Types and values from the remote runtime package
	runtime runtimeValues;

	// Runtime field indexes
	f runtimeIndexes;

	// Globals from the sys package (or from no package)
	sys struct {
		lessstack, goexit, newproc, deferproc *sym.TextSym;
	};
}

// NewProcess constructs a new remote process around a ptrace'd
// process, an architecture, and a symbol table.
func NewProcess(proc ptrace.Process, arch Arch, syms *sym.GoSymTable) *Process {
	p := &Process{
		Arch: arch,
		Process: proc,
		syms: syms,
		types: make(map[ptrace.Word] *remoteType),
	};

	// TODO(austin) Set p.frame if proc is stopped

	p.bootstrap();
	return p;
}

// NewProcessElf constructs a new remote process around a ptrace'd
// process and the process' ELF object.
func NewProcessElf(proc ptrace.Process, elf *sym.Elf) (*Process, os.Error) {
	syms, err := sym.ElfGoSyms(elf);
	if err != nil {
		return nil, err;
	}
	if syms == nil {
		return nil, FormatError("Failed to find symbol table");
	}
	var arch Arch;
	switch elf.Machine {
	case sym.ElfX86_64:
		arch = Amd64;
	default:
		return nil, UnknownArchitecture(elf.Machine);
	}
	return NewProcess(proc, arch, syms), nil;
}

// bootstrap constructs the runtime structure of a remote process.
func (p *Process) bootstrap() {
	// Manually construct runtime types
	p.runtime.String = newManualType(eval.TypeOfNative(rt1String{}), p.Arch);
	p.runtime.Slice = newManualType(eval.TypeOfNative(rt1Slice{}), p.Arch);
	p.runtime.Eface = newManualType(eval.TypeOfNative(rt1Eface{}), p.Arch);

	p.runtime.Type = newManualType(eval.TypeOfNative(rt1Type{}), p.Arch);
	p.runtime.CommonType = newManualType(eval.TypeOfNative(rt1CommonType{}), p.Arch);
	p.runtime.UncommonType = newManualType(eval.TypeOfNative(rt1UncommonType{}), p.Arch);
	p.runtime.StructField = newManualType(eval.TypeOfNative(rt1StructField{}), p.Arch);
	p.runtime.StructType = newManualType(eval.TypeOfNative(rt1StructType{}), p.Arch);
	p.runtime.PtrType = newManualType(eval.TypeOfNative(rt1PtrType{}), p.Arch);
	p.runtime.ArrayType = newManualType(eval.TypeOfNative(rt1ArrayType{}), p.Arch);
	p.runtime.SliceType = newManualType(eval.TypeOfNative(rt1SliceType{}), p.Arch);

	p.runtime.Stktop = newManualType(eval.TypeOfNative(rt1Stktop{}), p.Arch);
	p.runtime.Gobuf = newManualType(eval.TypeOfNative(rt1Gobuf{}), p.Arch);
	p.runtime.G = newManualType(eval.TypeOfNative(rt1G{}), p.Arch);

	// Get addresses of type·*runtime.XType for discrimination.
	rtv := reflect.Indirect(reflect.NewValue(&p.runtime)).(*reflect.StructValue);
	rtvt := rtv.Type().(*reflect.StructType);
	for i := 0; i < rtv.NumField(); i++ {
		n := rtvt.Field(i).Name;
		if n[0] != 'P' || n[1] < 'A' || n[1] > 'Z' {
			continue;
		}
		sym := p.syms.SymFromName("type·*runtime." + n[1:len(n)]);
		if sym == nil {
			continue;
		}
		rtv.Field(i).(*reflect.Uint64Value).Set(sym.Common().Value);
	}

	// Get runtime field indexes
	fillRuntimeIndexes(&p.runtime, &p.f);

	// Fill G status
	p.runtime.runtimeGStatus = rt1GStatus;

	// Get globals
	globalFn := func(name string) *sym.TextSym {
		if sym, ok := p.syms.SymFromName(name).(*sym.TextSym); ok {
			return sym;
		}
		return nil;
	};
	p.sys.lessstack = globalFn("sys·lessstack");
	p.sys.goexit = globalFn("goexit");
	p.sys.newproc = globalFn("sys·newproc");
	p.sys.deferproc = globalFn("sys·deferproc");
}

func (p *Process) someStoppedThread() ptrace.Thread {
	if p.threadCache != nil {
		if _, err := p.threadCache.Stopped(); err == nil {
			return p.threadCache;
		}
	}

	for _, t := range p.Threads() {
		if _, err := t.Stopped(); err == nil {
			p.threadCache = t;
			return t;
		}
	}
	return nil;
}

func (p *Process) Peek(addr ptrace.Word, out []byte) (int, os.Error) {
	thr := p.someStoppedThread();
	if thr == nil {
		return 0, ProcessNotStopped{};
	}
	return thr.Peek(addr, out);
}

func (p *Process) Poke(addr ptrace.Word, b []byte) (int, os.Error) {
	thr := p.someStoppedThread();
	if thr == nil {
		return 0, ProcessNotStopped{};
	}
	return thr.Poke(addr, b);
}

func (p *Process) peekUintptr(addr ptrace.Word) ptrace.Word {
	return ptrace.Word(mkUintptr(remote{addr, p}).(remoteUint).Get());
}
