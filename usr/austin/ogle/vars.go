// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ogle

import (
	"eval";
	"log";
	"os";
	"ptrace";
	"sym";
)

/*
 * Remote frame pointers
 */

// A NotOnStack error occurs when attempting to access a variable in a
// remote frame where that remote frame is not on the current stack.
type NotOnStack struct {
	Fn *sym.TextSym;
	Goroutine *Goroutine;
}

func (e NotOnStack) String() string {
	return "function " + e.Fn.Name + " not on " + e.Goroutine.String() + "'s stack";
}

// A remoteFramePtr is an implementation of eval.PtrValue that
// represents a pointer to a function frame in a remote process.  When
// accessed, this locates the function on the current goroutine's
// stack and returns a structure containing the local variables of
// that function.
type remoteFramePtr struct {
	p *Process;
	fn *sym.TextSym;
	rt *remoteType;
}

func (v remoteFramePtr) String() string {
	// TODO(austin): This could be a really awesome string method
	return "<remote frame>";
}

func (v remoteFramePtr) Assign(t *eval.Thread, o eval.Value) {
	v.Set(t, o.(eval.PtrValue).Get(t));
}

func (v remoteFramePtr) Get(t *eval.Thread) eval.Value {
	g := v.p.curGoroutine;
	if g == nil || g.frame == nil {
		t.Abort(NoCurrentGoroutine{});
	}

	for f := g.frame; f != nil; f = f.aOuter(t) {
		if f.fn != v.fn {
			continue;
		}

		// TODO(austin): Register for shootdown with f
		return v.rt.mk(remote{f.fp, v.p});
	}

	t.Abort(NotOnStack{v.fn, g});
	panic();
}

func (v remoteFramePtr) Set(t *eval.Thread, x eval.Value) {
	// Theoretically this could be a static error.  If remote
	// packages were packages, remote frames could just be defined
	// as constants.
	t.Abort(ReadOnlyError("remote frames cannot be assigned to"));
}

/*
 * Remote packages
 */

// TODO(austin): Remote packages are implemented as structs right now,
// which has some weird consequences.  You can attempt to assign to a
// remote package.  It also produces terrible error messages.
// Ideally, these would actually be packages, but somehow first-class
// so they could be assigned to other names.

// A remotePackage is an implementation of eval.StructValue that
// represents a package in a remote process.  It's essentially a
// regular struct, except it cannot be assigned to.
type remotePackage struct {
	defs []eval.Value;
}

func (v remotePackage) String() string {
	return "<remote package>";
}

func (v remotePackage) Assign(t *eval.Thread, o eval.Value) {
	t.Abort(ReadOnlyError("remote packages cannot be assigned to"));
}

func (v remotePackage) Get(t *eval.Thread) eval.StructValue {
	return v;
}

func (v remotePackage) Field(t *eval.Thread, i int) eval.Value {
	return v.defs[i];
}

/*
 * Remote variables
 */

// populateWorld defines constants in the given world for each package
// in this process.  These packages are structs that, in turn, contain
// fields for each global and function in that package.
func (p *Process) populateWorld(w *eval.World) os.Error {
	type def struct {
		t eval.Type;
		v eval.Value;
	}
	packages := make(map[string] map[string] def);

	for _, s := range p.syms.Syms {
		sc := s.Common();
		if sc.ReceiverName() != "" {
			// TODO(austin)
			continue;
		}

		// Package
		pkgName := sc.PackageName();
		switch pkgName {
		case "", "type", "extratype", "string", "go":
			// "go" is really "go.string"
			continue;
		}
		pkg, ok := packages[pkgName];
		if !ok {
			pkg = make(map[string] def);
			packages[pkgName] = pkg;
		}

		// Symbol name
		name := sc.BaseName();
		if prev, ok := pkg[name]; ok {
			log.Stderrf("Multiple definitions of symbol %s", sc.Name);
			continue;
		}

		// Symbol type
		rt, err := p.typeOfSym(sc);
		if err != nil {
			return err;
		}

		// Definition
		switch sc.Type {
		case 'D', 'd', 'B', 'b':
			// Global variable
			if rt == nil {
				continue;
			}
			pkg[name] = def{rt.Type, rt.mk(remote{ptrace.Word(sc.Value), p})};

		case 'T', 't', 'L', 'l':
			// Function
			s := s.(*sym.TextSym);
			// TODO(austin): Ideally, this would *also* be
			// callable.  How does that interact with type
			// conversion syntax?
			rt, err := p.makeFrameType(s);
			if err != nil {
				return err;
			}
			pkg[name] = def{eval.NewPtrType(rt.Type), remoteFramePtr{p, s, rt}};
		}
	}

	// TODO(austin): Define remote types

	// Define packages
	for pkgName, defs := range packages {
		fields := make([]eval.StructField, len(defs));
		vals := make([]eval.Value, len(defs));
		i := 0;
		for name, def := range defs {
			fields[i].Name = name;
			fields[i].Type = def.t;
			vals[i] = def.v;
			i++;
		}
		pkgType := eval.NewStructType(fields);
		pkgVal := remotePackage{vals};

		err := w.DefineConst(pkgName, pkgType, pkgVal);
		if err != nil {
			log.Stderrf("while defining package %s: %v", pkgName, err);
		}
	}

	return nil;
}

// typeOfSym returns the type associated with a symbol.  If the symbol
// has no type, returns nil.
func (p *Process) typeOfSym(s *sym.CommonSym) (*remoteType, os.Error) {
	if s.GoType == 0 {
		return nil, nil;
	}
	addr := ptrace.Word(s.GoType);
	var rt *remoteType;
	err := try(func(a aborter) {
		rt = parseRemoteType(a, p.runtime.Type.mk(remote{addr, p}).(remoteStruct));
	});
	if err != nil {
		return nil, err;
	}
	return rt, nil;
}

// makeFrameType constructs a struct type for the frame of a function.
// The offsets in this struct type are such that the struct can be
// instantiated at this function's frame pointer.
func (p *Process) makeFrameType(s *sym.TextSym) (*remoteType, os.Error) {
	n := len(s.Params) + len(s.Locals);
	fields := make([]eval.StructField, n);
	layout := make([]remoteStructField, n);
	i := 0;

	// TODO(austin): There can be multiple locals/parameters with
	// the same name.  We probably need liveness information to do
	// anything about this.  Once we have that, perhaps we give
	// such fields interface{} type?  Or perhaps we disambiguate
	// the names with numbers.  Disambiguation is annoying for
	// things like "i", where there's an obvious right answer.

	for _, param := range s.Params {
		rt, err := p.typeOfSym(param.Common());
		if err != nil {
			return nil, err;
		}
		if rt == nil {
			//fmt.Printf(" (no type)\n");
			continue;
		}
		// TODO(austin): Why do local variables carry their
		// package name?
		fields[i].Name = param.BaseName();
		fields[i].Type = rt.Type;
		// Parameters have positive offsets from FP
		layout[i].offset = int(param.Value);
		layout[i].fieldType = rt;
		i++;
	}

	for _, local := range s.Locals {
		rt, err := p.typeOfSym(local.Common());
		if err != nil {
			return nil, err;
		}
		if rt == nil {
			continue;
		}
		fields[i].Name = local.BaseName();
		fields[i].Type = rt.Type;
		// Locals have negative offsets from FP - PtrSize
		layout[i].offset = -int(local.Value) - p.PtrSize();
		layout[i].fieldType = rt;
		i++;
	}

	fields = fields[0:i];
	layout = layout[0:i];
	t := eval.NewStructType(fields);
	mk := func(r remote) eval.Value { return remoteStruct{r, layout} };
	return &remoteType{t, 0, 0, mk}, nil;
}
