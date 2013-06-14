package ssa

// This file implements the CREATE phase of SSA construction.
// See builder.go for explanation.

import (
	"fmt"
	"go/ast"
	"go/token"
	"os"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/importer"
)

// BuilderMode is a bitmask of options for diagnostics and checking.
type BuilderMode uint

const (
	LogPackages          BuilderMode = 1 << iota // Dump package inventory to stderr
	LogFunctions                                 // Dump function SSA code to stderr
	LogSource                                    // Show source locations as SSA builder progresses
	SanityCheckFunctions                         // Perform sanity checking of function bodies
	NaiveForm                                    // Build naïve SSA form: don't replace local loads/stores with registers
	BuildSerially                                // Build packages serially, not in parallel.
)

// NewProgram returns a new SSA Program initially containing no
// packages.
//
// fset specifies the mapping from token positions to source location
// that will be used by all ASTs of this program.
//
// mode controls diagnostics and checking during SSA construction.
//
func NewProgram(fset *token.FileSet, mode BuilderMode) *Program {
	prog := &Program{
		Files:               fset,
		Packages:            make(map[string]*Package),
		packages:            make(map[*types.Package]*Package),
		Builtins:            make(map[types.Object]*Builtin),
		methodSets:          make(map[types.Type]MethodSet),
		concreteMethods:     make(map[*types.Func]*Function),
		indirectionWrappers: make(map[*Function]*Function),
		boundMethodWrappers: make(map[*Function]*Function),
		ifaceMethodWrappers: make(map[*types.Func]*Function),
		mode:                mode,
	}

	// Create Values for built-in functions.
	for i, n := 0, types.Universe.NumEntries(); i < n; i++ {
		if obj, ok := types.Universe.At(i).(*types.Func); ok {
			prog.Builtins[obj] = &Builtin{obj}
		}
	}

	return prog
}

// CreatePackages creates an SSA Package for each type-checker package
// held by imp.  All such packages must be error-free.
//
// The created packages may be accessed via the Program.Packages field.
//
// A package in the 'created' state has its Members mapping populated,
// but a subsequent call to Package.Build() or Program.BuildAll() is
// required to build SSA code for the bodies of its functions.
//
func (prog *Program) CreatePackages(imp *importer.Importer) {
	// TODO(adonovan): make this idempotent, so that a second call
	// to CreatePackages creates only the packages that appeared
	// in imp since the first.
	for path, info := range imp.Packages {
		createPackage(prog, path, info)
	}
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
		pkg.Members[name] = &Type{Object: obj}

	case *types.Const:
		pkg.Members[name] = &Constant{
			name:  name,
			Value: NewLiteral(obj.Val(), obj.Type()),
			pos:   obj.Pos(),
		}

	case *types.Var:
		spec, _ := syntax.(*ast.ValueSpec)
		g := &Global{
			Pkg:  pkg,
			name: name,
			typ:  pointer(obj.Type()), // address
			pos:  obj.Pos(),
			spec: spec,
		}
		pkg.values[obj] = g
		pkg.Members[name] = g

	case *types.Func:
		var fs *funcSyntax
		if decl, ok := syntax.(*ast.FuncDecl); ok {
			fs = &funcSyntax{
				recvField:    decl.Recv,
				paramFields:  decl.Type.Params,
				resultFields: decl.Type.Results,
				body:         decl.Body,
			}
		}
		sig := obj.Type().(*types.Signature)
		fn := &Function{
			name:      name,
			Signature: sig,
			pos:       obj.Pos(), // (iff syntax)
			Pkg:       pkg,
			Prog:      pkg.Prog,
			syntax:    fs,
		}
		if recv := sig.Recv(); recv == nil {
			// Function declaration.
			pkg.values[obj] = fn
			pkg.Members[name] = fn
		} else {
			// Method declaration.
			_, method := namedTypeMethodIndex(
				recv.Type().Deref().(*types.Named),
				MakeId(name, pkg.Types))
			pkg.Prog.concreteMethods[method] = fn
		}

	default: // (incl. *types.Package)
		panic(fmt.Sprintf("unexpected Object type: %T", obj))
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
						memberFromObject(pkg, pkg.objectOf(id), nil)
					}
				}
			}

		case token.VAR:
			for _, spec := range decl.Specs {
				for _, id := range spec.(*ast.ValueSpec).Names {
					if !isBlankIdent(id) {
						memberFromObject(pkg, pkg.objectOf(id), spec)
					}
				}
			}

		case token.TYPE:
			for _, spec := range decl.Specs {
				id := spec.(*ast.TypeSpec).Name
				if !isBlankIdent(id) {
					memberFromObject(pkg, pkg.objectOf(id), nil)
				}
			}
		}

	case *ast.FuncDecl:
		id := decl.Name
		if decl.Recv == nil && id.Name == "init" {
			if !pkg.Init.pos.IsValid() {
				pkg.Init.pos = decl.Name.Pos()
			}
			return // init blocks aren't functions
		}
		if !isBlankIdent(id) {
			memberFromObject(pkg, pkg.objectOf(id), decl)
		}
	}
}

// createPackage constructs an SSA Package from an error-free
// package described by info, and populates its Members mapping.
//
// The real work of building SSA form for each function is not done
// until a subsequent call to Package.Build().
//
func createPackage(prog *Program, importPath string, info *importer.PackageInfo) {
	p := &Package{
		Prog:    prog,
		Members: make(map[string]Member),
		values:  make(map[types.Object]Value),
		Types:   info.Pkg,
		info:    info, // transient (CREATE and BUILD phases)
	}

	// Add init() function (but not to Members since it can't be referenced).
	p.Init = &Function{
		name:      "init",
		Signature: new(types.Signature),
		Pkg:       p,
		Prog:      prog,
	}

	// CREATE phase.
	// Allocate all package members: vars, funcs and consts and types.
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
		scope := p.Types.Scope()
		for i, n := 0, scope.NumEntries(); i < n; i++ {
			memberFromObject(p, scope.At(i), nil)
		}
	}

	// Add initializer guard variable.
	initguard := &Global{
		Pkg:  p,
		name: "init$guard",
		typ:  pointer(tBool),
	}
	p.Members[initguard.Name()] = initguard

	if prog.mode&LogPackages != 0 {
		p.DumpTo(os.Stderr)
	}

	prog.Packages[importPath] = p
	prog.packages[p.Types] = p
}
