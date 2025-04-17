// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"fmt"
	"strconv"
	"sync"
)

// pkgMap maps a package path to a package.
var pkgMap = make(map[string]*Pkg)

type Pkg struct {
	Path    string // string literal used in import statement, e.g. "internal/runtime/sys"
	Name    string // package name, e.g. "sys"
	Prefix  string // escaped path for use in symbol table
	Syms    map[string]*Sym
	Pathsym *obj.LSym

	Direct bool // imported directly
}

// NewPkg returns a new Pkg for the given package path and name.
// Unless name is the empty string, if the package exists already,
// the existing package name and the provided name must match.
func NewPkg(path, name string) *Pkg {
	if p := pkgMap[path]; p != nil {
		if name != "" && p.Name != name {
			panic(fmt.Sprintf("conflicting package names %s and %s for path %q", p.Name, name, path))
		}
		return p
	}

	p := new(Pkg)
	p.Path = path
	p.Name = name
	if path == "go.shape" {
		// Don't escape "go.shape", since it's not needed (it's a builtin
		// package), and we don't want escape codes showing up in shape type
		// names, which also appear in names of function/method
		// instantiations.
		p.Prefix = path
	} else {
		p.Prefix = objabi.PathToPrefix(path)
	}
	p.Syms = make(map[string]*Sym)
	pkgMap[path] = p

	return p
}

func PkgMap() map[string]*Pkg {
	return pkgMap
}

var nopkg = &Pkg{
	Syms: make(map[string]*Sym),
}

func (pkg *Pkg) Lookup(name string) *Sym {
	s, _ := pkg.LookupOK(name)
	return s
}

// LookupOK looks up name in pkg and reports whether it previously existed.
func (pkg *Pkg) LookupOK(name string) (s *Sym, existed bool) {
	// TODO(gri) remove this check in favor of specialized lookup
	if pkg == nil {
		pkg = nopkg
	}
	if s := pkg.Syms[name]; s != nil {
		return s, true
	}

	s = &Sym{
		Name: name,
		Pkg:  pkg,
	}
	pkg.Syms[name] = s
	return s, false
}

func (pkg *Pkg) LookupBytes(name []byte) *Sym {
	// TODO(gri) remove this check in favor of specialized lookup
	if pkg == nil {
		pkg = nopkg
	}
	if s := pkg.Syms[string(name)]; s != nil {
		return s
	}
	str := InternString(name)
	return pkg.Lookup(str)
}

// LookupNum looks up the symbol starting with prefix and ending with
// the decimal n. If prefix is too long, LookupNum panics.
func (pkg *Pkg) LookupNum(prefix string, n int) *Sym {
	var buf [20]byte // plenty long enough for all current users
	copy(buf[:], prefix)
	b := strconv.AppendInt(buf[:len(prefix)], int64(n), 10)
	return pkg.LookupBytes(b)
}

// Selector looks up a selector identifier.
func (pkg *Pkg) Selector(name string) *Sym {
	if IsExported(name) {
		pkg = LocalPkg
	}
	return pkg.Lookup(name)
}

var (
	internedStringsmu sync.Mutex // protects internedStrings
	internedStrings   = map[string]string{}
)

func InternString(b []byte) string {
	internedStringsmu.Lock()
	s, ok := internedStrings[string(b)] // string(b) here doesn't allocate
	if !ok {
		s = string(b)
		internedStrings[s] = s
	}
	internedStringsmu.Unlock()
	return s
}
