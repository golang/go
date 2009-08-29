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

// A Process represents a remote attached process.
type Process struct {
	Arch;
	ptrace.Process;

	// The symbol table of this process
	syms *sym.GoSymTable;

	// Current thread
	thread ptrace.Thread;
	// Current frame, or nil if the current thread is not stopped
	frame *frame;

	// Types parsed from the remote process
	types map[ptrace.Word] *remoteType;

	// Types and values from the remote runtime package
	runtime runtimeValues;

	// Runtime field indexes
	f runtimeIndexes;
}

// NewProcess constructs a new remote process around a ptrace'd
// process, an architecture, and a symbol table.
func NewProcess(proc ptrace.Process, arch Arch, syms *sym.GoSymTable) *Process {
	p := &Process{
		Arch: arch,
		Process: proc,
		syms: syms,
		thread: proc.Threads()[0],
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

	// Get field indexes
	fillRuntimeIndexes(&p.runtime, &p.f);
}
