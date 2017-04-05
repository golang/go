// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import "cmd/internal/obj"

type Pkg struct {
	Name     string // package name, e.g. "sys"
	Path     string // string literal used in import statement, e.g. "runtime/internal/sys"
	Pathsym  *obj.LSym
	Prefix   string // escaped path for use in symbol table
	Imported bool   // export data of this package was parsed
	Direct   bool   // imported directly
	Syms     map[string]*Sym
}

var Nopkg = &Pkg{
	Syms: make(map[string]*Sym),
}

func (pkg *Pkg) Lookup(name string) *Sym {
	s, _ := pkg.LookupOK(name)
	return s
}

var InitSyms []*Sym

// LookupOK looks up name in pkg and reports whether it previously existed.
func (pkg *Pkg) LookupOK(name string) (s *Sym, existed bool) {
	if pkg == nil {
		pkg = Nopkg
	}
	if s := pkg.Syms[name]; s != nil {
		return s, true
	}

	s = &Sym{
		Name: name,
		Pkg:  pkg,
	}
	if name == "init" {
		InitSyms = append(InitSyms, s)
	}
	pkg.Syms[name] = s
	return s, false
}

func (pkg *Pkg) LookupBytes(name []byte) *Sym {
	if pkg == nil {
		pkg = Nopkg
	}
	if s := pkg.Syms[string(name)]; s != nil {
		return s
	}
	str := InternString(name)
	return pkg.Lookup(str)
}

var internedStrings = map[string]string{}

func InternString(b []byte) string {
	s, ok := internedStrings[string(b)] // string(b) here doesn't allocate
	if !ok {
		s = string(b)
		internedStrings[s] = s
	}
	return s
}
