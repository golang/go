// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file implements the CREATE phase of SSA construction.
// See builder.go for explanation.

import (
	"go/ast"
	"go/token"
	"os"
	"sync"

	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/types"
)

// BuilderMode is a bitmask of options for diagnostics and checking.
type BuilderMode uint

const (
	PrintPackages        BuilderMode = 1 << iota // Print package inventory to stdout
	PrintFunctions                               // Print function SSA code to stdout
	LogSource                                    // Log source locations as SSA builder progresses
	SanityCheckFunctions                         // Perform sanity checking of function bodies
	NaiveForm                                    // Build naïve SSA form: don't replace local loads/stores with registers
	BuildSerially                                // Build packages serially, not in parallel.
	GlobalDebug                                  // Enable debug info for all packages
	BareInits                                    // Build init functions without guards or calls to dependent inits
)

// Create returns a new SSA Program.  An SSA Package is created for
// each transitively error-free package of iprog.
//
// Code for bodies of functions is not built until Build() is called
// on the result.
//
// mode controls diagnostics and checking during SSA construction.
//
func Create(iprog *loader.Program, mode BuilderMode) *Program {
	prog := &Program{
		Fset:     iprog.Fset,
		imported: make(map[string]*Package),
		packages: make(map[*types.Package]*Package),
		thunks:   make(map[selectionKey]*Function),
		bounds:   make(map[*types.Func]*Function),
		mode:     mode,
	}

	for _, info := range iprog.AllPackages {
		// TODO(adonovan): relax this constraint if the
		// program contains only "soft" errors.
		if info.TransitivelyErrorFree {
			prog.CreatePackage(info)
		}
	}

	return prog
}

// memberFromObject populates package pkg with a member for the
// typechecker object obj.
//
// For objects from Go source code, syntax is the associated syntax
// tree (for funcs and vars only); it will be used during the build
// phase.
//
func memberFromObject(pkg *Package, obj types.Object, syntax ast.Node) {
	name := obj.Name()
	switch obj := obj.(type) {
	case *types.TypeName:
		pkg.Members[name] = &Type{
			object: obj,
			pkg:    pkg,
		}

	case *types.Const:
		c := &NamedConst{
			object: obj,
			Value:  NewConst(obj.Val(), obj.Type()),
			pkg:    pkg,
		}
		pkg.values[obj] = c.Value
		pkg.Members[name] = c

	case *types.Var:
		g := &Global{
			Pkg:    pkg,
			name:   name,
			object: obj,
			typ:    types.NewPointer(obj.Type()), // address
			pos:    obj.Pos(),
		}
		pkg.values[obj] = g
		pkg.Members[name] = g

	case *types.Func:
		fn := &Function{
			name:      name,
			object:    obj,
			Signature: obj.Type().(*types.Signature),
			syntax:    syntax,
			pos:       obj.Pos(),
			Pkg:       pkg,
			Prog:      pkg.Prog,
		}
		if syntax == nil {
			fn.Synthetic = "loaded from gc object file"
		}

		pkg.values[obj] = fn
		if fn.Signature.Recv() == nil {
			pkg.Members[name] = fn // package-level function
		}

	default: // (incl. *types.Package)
		panic("unexpected Object type: " + obj.String())
	}
}

// membersFromDecl populates package pkg with members for each
// typechecker object (var, func, const or type) associated with the
// specified decl.
//
func membersFromDecl(pkg *Package, decl ast.Decl) {
	switch decl := decl.(type) {
	case *ast.GenDecl: // import, const, type or var
		switch decl.Tok {
		case token.CONST:
			for _, spec := range decl.Specs {
				for _, id := range spec.(*ast.ValueSpec).Names {
					if !isBlankIdent(id) {
						memberFromObject(pkg, pkg.info.Defs[id], nil)
					}
				}
			}

		case token.VAR:
			for _, spec := range decl.Specs {
				for _, id := range spec.(*ast.ValueSpec).Names {
					if !isBlankIdent(id) {
						memberFromObject(pkg, pkg.info.Defs[id], spec)
					}
				}
			}

		case token.TYPE:
			for _, spec := range decl.Specs {
				id := spec.(*ast.TypeSpec).Name
				if !isBlankIdent(id) {
					memberFromObject(pkg, pkg.info.Defs[id], nil)
				}
			}
		}

	case *ast.FuncDecl:
		id := decl.Name
		if decl.Recv == nil && id.Name == "init" {
			return // no object
		}
		if !isBlankIdent(id) {
			memberFromObject(pkg, pkg.info.Defs[id], decl)
		}
	}
}

// CreatePackage constructs and returns an SSA Package from an
// error-free package described by info, and populates its Members
// mapping.
//
// Repeated calls with the same info return the same Package.
//
// The real work of building SSA form for each function is not done
// until a subsequent call to Package.Build().
//
func (prog *Program) CreatePackage(info *loader.PackageInfo) *Package {
	if p := prog.packages[info.Pkg]; p != nil {
		return p // already loaded
	}

	p := &Package{
		Prog:    prog,
		Members: make(map[string]Member),
		values:  make(map[types.Object]Value),
		Object:  info.Pkg,
		info:    info, // transient (CREATE and BUILD phases)
	}

	// Add init() function.
	p.init = &Function{
		name:      "init",
		Signature: new(types.Signature),
		Synthetic: "package initializer",
		Pkg:       p,
		Prog:      prog,
	}
	p.Members[p.init.name] = p.init

	// CREATE phase.
	// Allocate all package members: vars, funcs, consts and types.
	if len(info.Files) > 0 {
		// Go source package.
		for _, file := range info.Files {
			for _, decl := range file.Decls {
				membersFromDecl(p, decl)
			}
		}
	} else {
		// GC-compiled binary package.
		// No code.
		// No position information.
		scope := p.Object.Scope()
		for _, name := range scope.Names() {
			obj := scope.Lookup(name)
			memberFromObject(p, obj, nil)
			if obj, ok := obj.(*types.TypeName); ok {
				named := obj.Type().(*types.Named)
				for i, n := 0, named.NumMethods(); i < n; i++ {
					memberFromObject(p, named.Method(i), nil)
				}
			}
		}
	}

	if prog.mode&BareInits == 0 {
		// Add initializer guard variable.
		initguard := &Global{
			Pkg:  p,
			name: "init$guard",
			typ:  types.NewPointer(tBool),
		}
		p.Members[initguard.Name()] = initguard
	}

	if prog.mode&GlobalDebug != 0 {
		p.SetDebugMode(true)
	}

	if prog.mode&PrintPackages != 0 {
		printMu.Lock()
		p.WriteTo(os.Stdout)
		printMu.Unlock()
	}

	if info.Importable {
		prog.imported[info.Pkg.Path()] = p
	}
	prog.packages[p.Object] = p

	return p
}

// printMu serializes printing of Packages/Functions to stdout
var printMu sync.Mutex

// AllPackages returns a new slice containing all packages in the
// program prog in unspecified order.
//
func (prog *Program) AllPackages() []*Package {
	pkgs := make([]*Package, 0, len(prog.packages))
	for _, pkg := range prog.packages {
		pkgs = append(pkgs, pkg)
	}
	return pkgs
}

// ImportedPackage returns the importable SSA Package whose import
// path is path, or nil if no such SSA package has been created.
//
// Not all packages are importable.  For example, no import
// declaration can resolve to the x_test package created by 'go test'
// or the ad-hoc main package created 'go build foo.go'.
//
func (prog *Program) ImportedPackage(path string) *Package {
	return prog.imported[path]
}
