// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package doc extracts source code documentation from a Go AST.
package doc

import (
	"fmt"
	"go/ast"
	"go/doc/comment"
	"go/token"
	"strings"
)

// Package is the documentation for an entire package.
type Package struct {
	Doc        string
	Name       string
	ImportPath string
	Imports    []string
	Filenames  []string
	Notes      map[string][]*Note

	// Deprecated: For backward compatibility Bugs is still populated,
	// but all new code should use Notes instead.
	Bugs []string

	// declarations
	Consts []*Value
	Types  []*Type
	Vars   []*Value
	Funcs  []*Func

	// Examples is a sorted list of examples associated with
	// the package. Examples are extracted from _test.go files
	// provided to NewFromFiles.
	Examples []*Example

	importByName map[string]string
	syms         map[string]bool
}

// Value is the documentation for a (possibly grouped) var or const declaration.
type Value struct {
	Doc   string
	Names []string // var or const names in declaration order
	Decl  *ast.GenDecl

	order int
}

// Type is the documentation for a type declaration.
type Type struct {
	Doc  string
	Name string
	Decl *ast.GenDecl

	// associated declarations
	Consts  []*Value // sorted list of constants of (mostly) this type
	Vars    []*Value // sorted list of variables of (mostly) this type
	Funcs   []*Func  // sorted list of functions returning this type
	Methods []*Func  // sorted list of methods (including embedded ones) of this type

	// Examples is a sorted list of examples associated with
	// this type. Examples are extracted from _test.go files
	// provided to NewFromFiles.
	Examples []*Example
}

// Func is the documentation for a func declaration.
type Func struct {
	Doc  string
	Name string
	Decl *ast.FuncDecl

	// methods
	// (for functions, these fields have the respective zero value)
	Recv  string // actual   receiver "T" or "*T" possibly followed by type parameters [P1, ..., Pn]
	Orig  string // original receiver "T" or "*T"
	Level int    // embedding level; 0 means not embedded

	// Examples is a sorted list of examples associated with this
	// function or method. Examples are extracted from _test.go files
	// provided to NewFromFiles.
	Examples []*Example
}

// A Note represents a marked comment starting with "MARKER(uid): note body".
// Any note with a marker of 2 or more upper case [A-Z] letters and a uid of
// at least one character is recognized. The ":" following the uid is optional.
// Notes are collected in the Package.Notes map indexed by the notes marker.
type Note struct {
	Pos, End token.Pos // position range of the comment containing the marker
	UID      string    // uid found with the marker
	Body     string    // note body text
}

// Mode values control the operation of [New] and [NewFromFiles].
type Mode int

const (
	// AllDecls says to extract documentation for all package-level
	// declarations, not just exported ones.
	AllDecls Mode = 1 << iota

	// AllMethods says to show all embedded methods, not just the ones of
	// invisible (unexported) anonymous fields.
	AllMethods

	// PreserveAST says to leave the AST unmodified. Originally, pieces of
	// the AST such as function bodies were nil-ed out to save memory in
	// godoc, but not all programs want that behavior.
	PreserveAST
)

// New computes the package documentation for the given package AST.
// New takes ownership of the AST pkg and may edit or overwrite it.
// To have the [Examples] fields populated, use [NewFromFiles] and include
// the package's _test.go files.
func New(pkg *ast.Package, importPath string, mode Mode) *Package {
	var r reader
	r.readPackage(pkg, mode)
	r.computeMethodSets()
	r.cleanupTypes()
	p := &Package{
		Doc:        r.doc,
		Name:       pkg.Name,
		ImportPath: importPath,
		Imports:    sortedKeys(r.imports),
		Filenames:  r.filenames,
		Notes:      r.notes,
		Bugs:       noteBodies(r.notes["BUG"]),
		Consts:     sortedValues(r.values, token.CONST),
		Types:      sortedTypes(r.types, mode&AllMethods != 0),
		Vars:       sortedValues(r.values, token.VAR),
		Funcs:      sortedFuncs(r.funcs, true),

		importByName: r.importByName,
		syms:         make(map[string]bool),
	}

	p.collectValues(p.Consts)
	p.collectValues(p.Vars)
	p.collectTypes(p.Types)
	p.collectFuncs(p.Funcs)

	return p
}

func (p *Package) collectValues(values []*Value) {
	for _, v := range values {
		for _, name := range v.Names {
			p.syms[name] = true
		}
	}
}

func (p *Package) collectTypes(types []*Type) {
	for _, t := range types {
		if p.syms[t.Name] {
			// Shouldn't be any cycles but stop just in case.
			continue
		}
		p.syms[t.Name] = true
		p.collectValues(t.Consts)
		p.collectValues(t.Vars)
		p.collectFuncs(t.Funcs)
		p.collectFuncs(t.Methods)
	}
}

func (p *Package) collectFuncs(funcs []*Func) {
	for _, f := range funcs {
		if f.Recv != "" {
			r := strings.TrimPrefix(f.Recv, "*")
			if i := strings.IndexByte(r, '['); i >= 0 {
				r = r[:i] // remove type parameters
			}
			p.syms[r+"."+f.Name] = true
		} else {
			p.syms[f.Name] = true
		}
	}
}

// NewFromFiles computes documentation for a package.
//
// The package is specified by a list of *ast.Files and corresponding
// file set, which must not be nil.
// NewFromFiles uses all provided files when computing documentation,
// so it is the caller's responsibility to provide only the files that
// match the desired build context. "go/build".Context.MatchFile can
// be used for determining whether a file matches a build context with
// the desired GOOS and GOARCH values, and other build constraints.
// The import path of the package is specified by importPath.
//
// Examples found in _test.go files are associated with the corresponding
// type, function, method, or the package, based on their name.
// If the example has a suffix in its name, it is set in the
// [Example.Suffix] field. [Examples] with malformed names are skipped.
//
// Optionally, a single extra argument of type [Mode] can be provided to
// control low-level aspects of the documentation extraction behavior.
//
// NewFromFiles takes ownership of the AST files and may edit them,
// unless the PreserveAST Mode bit is on.
func NewFromFiles(fset *token.FileSet, files []*ast.File, importPath string, opts ...any) (*Package, error) {
	// Check for invalid API usage.
	if fset == nil {
		panic(fmt.Errorf("doc.NewFromFiles: no token.FileSet provided (fset == nil)"))
	}
	var mode Mode
	switch len(opts) { // There can only be 0 or 1 options, so a simple switch works for now.
	case 0:
		// Nothing to do.
	case 1:
		m, ok := opts[0].(Mode)
		if !ok {
			panic(fmt.Errorf("doc.NewFromFiles: option argument type must be doc.Mode"))
		}
		mode = m
	default:
		panic(fmt.Errorf("doc.NewFromFiles: there must not be more than 1 option argument"))
	}

	// Collect .go and _test.go files.
	var (
		goFiles     = make(map[string]*ast.File)
		testGoFiles []*ast.File
	)
	for i := range files {
		f := fset.File(files[i].Pos())
		if f == nil {
			return nil, fmt.Errorf("file files[%d] is not found in the provided file set", i)
		}
		switch name := f.Name(); {
		case strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, "_test.go"):
			goFiles[name] = files[i]
		case strings.HasSuffix(name, "_test.go"):
			testGoFiles = append(testGoFiles, files[i])
		default:
			return nil, fmt.Errorf("file files[%d] filename %q does not have a .go extension", i, name)
		}
	}

	// TODO(dmitshur,gri): A relatively high level call to ast.NewPackage with a simpleImporter
	// ast.Importer implementation is made below. It might be possible to short-circuit and simplify.

	// Compute package documentation.
	pkg, _ := ast.NewPackage(fset, goFiles, simpleImporter, nil) // Ignore errors that can happen due to unresolved identifiers.
	p := New(pkg, importPath, mode)
	classifyExamples(p, Examples(testGoFiles...))
	return p, nil
}

// simpleImporter returns a (dummy) package object named by the last path
// component of the provided package path (as is the convention for packages).
// This is sufficient to resolve package identifiers without doing an actual
// import. It never returns an error.
func simpleImporter(imports map[string]*ast.Object, path string) (*ast.Object, error) {
	pkg := imports[path]
	if pkg == nil {
		// note that strings.LastIndex returns -1 if there is no "/"
		pkg = ast.NewObj(ast.Pkg, path[strings.LastIndex(path, "/")+1:])
		pkg.Data = ast.NewScope(nil) // required by ast.NewPackage for dot-import
		imports[path] = pkg
	}
	return pkg, nil
}

// lookupSym reports whether the package has a given symbol or method.
//
// If recv == "", HasSym reports whether the package has a top-level
// const, func, type, or var named name.
//
// If recv != "", HasSym reports whether the package has a type
// named recv with a method named name.
func (p *Package) lookupSym(recv, name string) bool {
	if recv != "" {
		return p.syms[recv+"."+name]
	}
	return p.syms[name]
}

// lookupPackage returns the import path identified by name
// in the given package. If name uniquely identifies a single import,
// then lookupPackage returns that import.
// If multiple packages are imported as name, importPath returns "", false.
// Otherwise, if name is the name of p itself, importPath returns "", true,
// to signal a reference to p.
// Otherwise, importPath returns "", false.
func (p *Package) lookupPackage(name string) (importPath string, ok bool) {
	if path, ok := p.importByName[name]; ok {
		if path == "" {
			return "", false // multiple imports used the name
		}
		return path, true // found import
	}
	if p.Name == name {
		return "", true // allow reference to this package
	}
	return "", false // unknown name
}

// Parser returns a doc comment parser configured
// for parsing doc comments from package p.
// Each call returns a new parser, so that the caller may
// customize it before use.
func (p *Package) Parser() *comment.Parser {
	return &comment.Parser{
		LookupPackage: p.lookupPackage,
		LookupSym:     p.lookupSym,
	}
}

// Printer returns a doc comment printer configured
// for printing doc comments from package p.
// Each call returns a new printer, so that the caller may
// customize it before use.
func (p *Package) Printer() *comment.Printer {
	// No customization today, but having p.Printer()
	// gives us flexibility in the future, and it is convenient for callers.
	return &comment.Printer{}
}

// HTML returns formatted HTML for the doc comment text.
//
// To customize details of the HTML, use [Package.Printer]
// to obtain a [comment.Printer], and configure it
// before calling its HTML method.
func (p *Package) HTML(text string) []byte {
	return p.Printer().HTML(p.Parser().Parse(text))
}

// Markdown returns formatted Markdown for the doc comment text.
//
// To customize details of the Markdown, use [Package.Printer]
// to obtain a [comment.Printer], and configure it
// before calling its Markdown method.
func (p *Package) Markdown(text string) []byte {
	return p.Printer().Markdown(p.Parser().Parse(text))
}

// Text returns formatted text for the doc comment text,
// wrapped to 80 Unicode code points and using tabs for
// code block indentation.
//
// To customize details of the formatting, use [Package.Printer]
// to obtain a [comment.Printer], and configure it
// before calling its Text method.
func (p *Package) Text(text string) []byte {
	return p.Printer().Text(p.Parser().Parse(text))
}
