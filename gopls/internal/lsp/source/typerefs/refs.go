// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typerefs

import (
	"fmt"
	"go/ast"
	"go/token"
	"sort"

	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/typeparams"
)

const debug = false

// declNode holds information about a package-level declaration
// (or more than one with the same name, in ill-typed code).
//
// It is a node in the symbol reference graph, whose outgoing edges
// are of two kinds: intRefs and extRefs.
type declNode struct {
	name string
	rep  *declNode // canonical representative of this SCC (initially self)

	// outgoing graph edges
	extRefs      map[Ref]bool       // to imported symbols
	intRefs      map[*declNode]bool // to symbols in this package
	extRefsSlice []Ref              // sorted keys of extRefs; populated at the end

	// Tarjan's SCC algorithm
	index, lowlink int32 // Tarjan numbering
	scc            int32 // -ve => on stack; 0 => unvisited; +ve => node is root of a found SCC
}

// A Ref is a reference to an external (imported) symbol.
type Ref struct {
	PkgID source.PackageID
	Name  string
}

// Refs analyzes all referring identifiers in the ParsedGoFile syntax,
// constructs a reference graph, and uses it to compute the
// reachability from each exported symbol (keys of the result map) in
// the package to the set of exported symbols of directly imported
// packages (values of the result map).
//
// See the package documentation for more details as to what a ref does (and
// does not) represent.
//
// The resulting map may have multiple keys with the same (slice) value,
// if two package members reach the same set of external symbols.
//
// References are ordered by (package, name).
func Refs(pgfs []*source.ParsedGoFile, id source.PackageID, imports map[source.ImportPath]*source.Metadata) map[string][]Ref {
	// First pass: gather package-level names and create a declNode for each.
	//
	// In ill-typed code, there may be multiple declarations of the
	// same name; a single declInfo node will represent them all.
	decls := make(map[string]*declNode)
	addDecl := func(id *ast.Ident) {
		if name := id.Name; name != "_" && decls[name] == nil {
			node := &declNode{name: name}
			node.rep = node
			decls[name] = node
		}
	}
	for _, pgf := range pgfs {
		for _, d := range pgf.File.Decls {
			switch d := d.(type) {
			case *ast.GenDecl:
				switch d.Tok {
				case token.TYPE:
					for _, spec := range d.Specs {
						addDecl(spec.(*ast.TypeSpec).Name)
					}

				case token.VAR, token.CONST:
					for _, spec := range d.Specs {
						for _, ident := range spec.(*ast.ValueSpec).Names {
							addDecl(ident)
						}
					}
				}

			case *ast.FuncDecl:
				// non-method functions
				if d.Recv.NumFields() == 0 {
					addDecl(d.Name)
				}
			}
		}
	}

	// Second pass: process files to collect referring identifiers.
	for _, pgf := range pgfs {
		visitFile(pgf.File, imports, decls)
	}

	// Find the strong components of the declNode graph
	// using Tarjan's algorithm, and coalesce each component.
	//
	// (This is the first of several graph optimizations inspired
	// by the Hardekopf and Lin algorithm used by the pointer
	// analysis in golang.org/x/go/pointer/hvn.go.)
	tj := tarjan{index: 1}
	for _, decl := range decls {
		if decl.index == 0 { // unvisited
			tj.visit(decl)
		}
	}

	// Populate the result map with the reachability
	// of each exported package member.
	edges := make(map[string][]Ref)
	for name, decl := range decls {
		if !ast.IsExported(name) {
			continue
		}

		// Many decls may have the same representative.
		// They will share (alias) the same result slice.
		decl = decl.find()
		if decl.extRefsSlice == nil {
			refs := make([]Ref, 0, len(decl.extRefs))
			for ref := range decl.extRefs {
				refs = append(refs, ref)
			}
			sort.Slice(refs, func(i, j int) bool {
				x, y := refs[i], refs[j]
				if x.PkgID != y.PkgID {
					return x.PkgID < y.PkgID
				}
				return x.Name < y.Name
			})
			decl.extRefsSlice = refs
		}
		if len(decl.extRefsSlice) > 0 {
			edges[name] = decl.extRefsSlice
		}
	}

	if trace {
		fmt.Printf("%s\n", id)
		var names []string
		for name := range edges {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			fmt.Printf("\t-> %s\n", name)
			// Group symbols by package.
			var prevID source.PackageID
			for _, ref := range edges[name] {
				if ref.PkgID != prevID {
					prevID = ref.PkgID
					fmt.Printf("\t\t-> %s:", ref.PkgID)
				}
				fmt.Printf(" %s", ref.Name)
			}
			fmt.Println()
		}
	}

	return edges
}

// visitFile inspects the file syntax for referring identifiers, and
// populates the internal and external references of decls.
func visitFile(file *ast.File, imports map[source.ImportPath]*source.Metadata, decls map[string]*declNode) {
	// Import information for this file. Multiple packages
	// may be referenced by a given name in the presence
	// of type errors (or multiple dot imports, which are
	// keyed by ".").
	fileImports := make(map[string][]source.PackageID)

	// importEdge records a reference from decl to an imported symbol
	// (pkgname.name). The package name may be ".".
	importEdge := func(decl *declNode, pkgname, name string) {
		if token.IsExported(name) {
			for _, depID := range fileImports[pkgname] {
				if decl.extRefs == nil {
					decl.extRefs = make(map[Ref]bool)
				}
				decl.extRefs[Ref{depID, name}] = true
			}
		}
	}

	// visit finds refs within node and builds edges from fromId's decl.
	// References to the type parameters are ignored.
	visit := func(fromId *ast.Ident, node ast.Node, tparams map[string]bool) {
		if fromId.Name == "_" {
			return
		}
		from := decls[fromId.Name]

		// Visit each reference to name or name.sel.
		visitDeclOrSpec(node, func(name, sel string) {
			// Ignore references to type parameters.
			if tparams[name] {
				return
			}

			// If name is declared in the package scope,
			// record an edge whether or not sel is empty.
			// A field or method selector may affect the
			// type of the current decl via initializers:
			//
			//  package p
			//  var x = y.F
			//  var y = struct{ F int }{}
			if to, ok := decls[name]; ok {
				if from.intRefs == nil {
					from.intRefs = make(map[*declNode]bool)
				}
				from.intRefs[to] = true

			} else {
				// Only record an edge to dot-imported packages
				// if there was no edge to a local name.
				// This assumes that there are no duplicate declarations.
				// We conservatively, assume that this name comes from
				// every dot-imported package.
				importEdge(from, ".", name)
			}

			// Record an edge to an import if it matches the name, even if that
			// name collides with a package level name. Unlike the case of dotted
			// imports, we know the package is invalid here, and choose to fail
			// conservatively.
			if sel != "" {
				importEdge(from, name, sel)
			}
		})
	}

	// Visit the declarations and gather reference edges.
	for _, d := range file.Decls {
		switch d := d.(type) {
		case *ast.GenDecl:
			switch d.Tok {
			case token.IMPORT:
				// Record local import names for this file.
				for _, spec := range d.Specs {
					spec := spec.(*ast.ImportSpec)
					path := source.UnquoteImportPath(spec)
					if path == "" {
						continue
					}
					dep := imports[path]
					if dep == nil {
						// Note here that we don't try to "guess"
						// the name of an import based on e.g.
						// its importPath. Doing so would only
						// result in edges that don't go anywhere.
						continue
					}
					name := string(dep.Name)
					if spec.Name != nil {
						if spec.Name.Name == "_" {
							continue
						}
						name = spec.Name.Name // possibly "."
					}
					fileImports[name] = append(fileImports[name], dep.ID)
				}

			case token.TYPE:
				for _, spec := range d.Specs {
					spec := spec.(*ast.TypeSpec)
					tparams := tparamsMap(typeparams.ForTypeSpec(spec))
					visit(spec.Name, spec, tparams)
				}

			case token.VAR, token.CONST:
				for _, spec := range d.Specs {
					spec := spec.(*ast.ValueSpec)
					for _, name := range spec.Names {
						visit(name, spec, nil)
					}
				}
			}

		case *ast.FuncDecl:
			// This check for NumFields() > 0 is consistent with go/types,
			// which reports an error but treats the declaration like a
			// normal function when Recv is non-nil but empty
			// (as in func () f()).
			if d.Recv.NumFields() > 0 {
				// Method. Associate it with the receiver.
				_, id, typeParams := unpackRecv(d.Recv.List[0].Type)
				var tparams map[string]bool
				if len(typeParams) > 0 {
					tparams = make(map[string]bool)
					for _, tparam := range typeParams {
						if tparam.Name != "_" {
							tparams[tparam.Name] = true
						}
					}
				}
				visit(id, d, tparams)
			} else {
				// Non-method.
				tparams := tparamsMap(typeparams.ForFuncType(d.Type))
				visit(d.Name, d, tparams)
			}
		}
	}
}

// tparamsMap returns a set recording each name declared by the provided field
// list. It so happens that we only care about names declared by type parameter
// lists.
func tparamsMap(tparams *ast.FieldList) map[string]bool {
	if tparams == nil || len(tparams.List) == 0 {
		return nil
	}
	m := make(map[string]bool)
	for _, f := range tparams.List {
		for _, name := range f.Names {
			if name.Name != "_" {
				m[name.Name] = true
			}
		}
	}
	return m
}

// A refVisitor visits referring identifiers and dotted identifiers.
//
// For a referring identifier I, name="I" and sel="". For a dotted identifier
// q.I, name="q" and sel="I".
type refVisitor = func(name, sel string)

// visitDeclOrSpec visits referring idents or dotted idents that may affect
// the type of the declaration at the given node, which must be an ast.Decl or
// ast.Spec.
func visitDeclOrSpec(node ast.Node, f refVisitor) {
	// Declarations
	switch n := node.(type) {
	// ImportSpecs should not appear here, and will panic in the default case.

	case *ast.ValueSpec:
		// Skip Doc, Names, Comments, which do not affect the decl type.
		// Initializers only affect the type of a value spec if the type is unset.
		if n.Type != nil {
			visitExpr(n.Type, f)
		} else { // only need to walk expr list if type is nil
			visitExprList(n.Values, f)
		}

	case *ast.TypeSpec:
		// Skip Doc, Name, and Comment, which do not affect the decl type.
		if tparams := typeparams.ForTypeSpec(n); tparams != nil {
			visitFieldList(tparams, f)
		}
		visitExpr(n.Type, f)

	case *ast.BadDecl:
		// nothing to do

	// We should not reach here with a GenDecl, so panic below in the default case.

	case *ast.FuncDecl:
		// Skip Doc, Name, and Body, which do not affect the type.
		// Recv is handled by Refs: methods are associated with their type.
		visitExpr(n.Type, f)

	default:
		panic(fmt.Sprintf("unexpected node type %T", node))
	}
}

// visitExpr visits referring idents and dotted idents that may affect the
// type of expr.
//
// visitExpr can't reliably distinguish a dotted ident pkg.X from a
// selection expr.f or T.method.
func visitExpr(expr ast.Expr, f refVisitor) {
	switch n := expr.(type) {
	// These four cases account for about two thirds of all nodes,
	// so we place them first to shorten the common control paths.
	// (See go.dev/cl/480915.)
	case *ast.Ident:
		f(n.Name, "")

	case *ast.BasicLit:
		// nothing to do

	case *ast.SelectorExpr:
		if ident, ok := n.X.(*ast.Ident); ok {
			f(ident.Name, n.Sel.Name)
		} else {
			visitExpr(n.X, f)
			// Skip n.Sel as we don't care about which field or method is selected,
			// as we'll have recorded an edge to all declarations relevant to the
			// receiver type via visiting n.X above.
		}

	case *ast.CallExpr:
		visitExpr(n.Fun, f)
		visitExprList(n.Args, f) // args affect types for unsafe.Sizeof or builtins or generics

	// Expressions
	case *ast.Ellipsis:
		if n.Elt != nil {
			visitExpr(n.Elt, f)
		}

	case *ast.FuncLit:
		visitExpr(n.Type, f)
		// Skip Body, which does not affect the type.

	case *ast.CompositeLit:
		if n.Type != nil {
			visitExpr(n.Type, f)
		}
		// Skip Elts, which do not affect the type.

	case *ast.ParenExpr:
		visitExpr(n.X, f)

	case *ast.IndexExpr:
		visitExpr(n.X, f)
		visitExpr(n.Index, f) // may affect type for instantiations

	case *typeparams.IndexListExpr:
		visitExpr(n.X, f)
		for _, index := range n.Indices {
			visitExpr(index, f) // may affect the type for instantiations
		}

	case *ast.SliceExpr:
		visitExpr(n.X, f)
		// skip Low, High, and Max, which do not affect type.

	case *ast.TypeAssertExpr:
		// Skip X, as it doesn't actually affect the resulting type of the type
		// assertion.
		if n.Type != nil {
			visitExpr(n.Type, f)
		}

	case *ast.StarExpr:
		visitExpr(n.X, f)

	case *ast.UnaryExpr:
		visitExpr(n.X, f)

	case *ast.BinaryExpr:
		visitExpr(n.X, f)
		visitExpr(n.Y, f)

	case *ast.KeyValueExpr:
		panic("unreachable") // unreachable, as we don't descend into elts of composite lits.

	case *ast.ArrayType:
		if n.Len != nil {
			visitExpr(n.Len, f)
		}
		visitExpr(n.Elt, f)

	case *ast.StructType:
		visitFieldList(n.Fields, f)

	case *ast.FuncType:
		if tparams := typeparams.ForFuncType(n); tparams != nil {
			visitFieldList(tparams, f)
		}
		if n.Params != nil {
			visitFieldList(n.Params, f)
		}
		if n.Results != nil {
			visitFieldList(n.Results, f)
		}

	case *ast.InterfaceType:
		visitFieldList(n.Methods, f)

	case *ast.MapType:
		visitExpr(n.Key, f)
		visitExpr(n.Value, f)

	case *ast.ChanType:
		visitExpr(n.Value, f)

	case *ast.BadExpr:
		// nothing to do

	default:
		panic(fmt.Sprintf("ast.Walk: unexpected node type %T", n))
	}
}

func visitExprList(list []ast.Expr, f refVisitor) {
	for _, x := range list {
		visitExpr(x, f)
	}
}

func visitFieldList(n *ast.FieldList, f refVisitor) {
	for _, field := range n.List {
		visitExpr(field.Type, f)
	}
}

// Copied (with modifications) from go/types.
func unpackRecv(rtyp ast.Expr) (ptr bool, rname *ast.Ident, tparams []*ast.Ident) {
L: // unpack receiver type
	// This accepts invalid receivers such as ***T and does not
	// work for other invalid receivers, but we don't care. The
	// validity of receiver expressions is checked elsewhere.
	for {
		switch t := rtyp.(type) {
		case *ast.ParenExpr:
			rtyp = t.X
		case *ast.StarExpr:
			ptr = true
			rtyp = t.X
		default:
			break L
		}
	}

	// unpack type parameters, if any
	switch rtyp.(type) {
	case *ast.IndexExpr, *typeparams.IndexListExpr:
		var indices []ast.Expr
		rtyp, _, indices, _ = typeparams.UnpackIndexExpr(rtyp)
		for _, arg := range indices {
			var par *ast.Ident
			switch arg := arg.(type) {
			case *ast.Ident:
				par = arg
			default:
				// ignore errors
			}
			if par == nil {
				par = &ast.Ident{NamePos: arg.Pos(), Name: "_"}
			}
			tparams = append(tparams, par)
		}
	}

	// unpack receiver name
	if name, _ := rtyp.(*ast.Ident); name != nil {
		rname = name
	}

	return
}

// -- strong component graph construction (plundered from go/pointer) --

type tarjan struct {
	index int32
	stack []*declNode
}

// visit implements the depth-first search of Tarjan's SCC algorithm.
// Precondition: x is canonical.
func (tj *tarjan) visit(x *declNode) {
	checkCanonical(x)
	x.index = tj.index
	x.lowlink = tj.index
	tj.index++

	tj.stack = append(tj.stack, x) // push
	assert(x.scc == 0, "node revisited")
	x.scc = -1

	for y := range x.intRefs {
		// Loop invariant: x is canonical.

		y := y.find()

		if x == y {
			continue // nodes already coalesced
		}

		switch {
		case y.scc > 0:
			// y is already a collapsed SCC

		case y.scc < 0:
			// y is on the stack, and thus in the current SCC.
			if y.index < x.lowlink {
				x.lowlink = y.index
			}

		default:
			// y is unvisited; visit it now.
			tj.visit(y)
			// Note: x and y are now non-canonical.

			x = x.find()

			if y.lowlink < x.lowlink {
				x.lowlink = y.lowlink
			}
		}
	}
	checkCanonical(x)

	// Is x the root of an SCC?
	if x.lowlink == x.index {
		// Coalesce all nodes in the SCC.
		for {
			// Pop y from stack.
			i := len(tj.stack) - 1
			y := tj.stack[i]
			tj.stack = tj.stack[:i]

			checkCanonical(x)
			checkCanonical(y)

			if x == y {
				// SCC is complete.
				x.scc = 1
				labelSCC(x)
				break
			}
			coalesce(x, y)
		}
	}
}

// labelSCC computes an equivalence label for a new SC node.
// Precondition: x is canonical.
func labelSCC(x *declNode) {
	// Compute union of extrefs over edges.
	// Find all extRefs coming in to the coalesced SCC node.
	for y := range x.intRefs {
		y := y.find()
		if y == x {
			continue // already coalesced
		}
		for z := range y.extRefs {
			if x.extRefs == nil {
				x.extRefs = make(map[Ref]bool)
			}
			x.extRefs[z] = true // extRefs: x U= y
		}
	}

	// TODO(adonovan): opt: implement PE algorithm here.
}

// coalesce combines two nodes in the strong component graph.
// Precondition: x and y are canonical.
func coalesce(x, y *declNode) {
	// x becomes y's canonical representative.
	y.rep = x

	// x accumulates y's internal references.
	for z := range y.intRefs {
		x.intRefs[z] = true
	}
	y.intRefs = nil

	// x accumulates y's external references.
	for z := range y.extRefs {
		if x.extRefs == nil {
			x.extRefs = make(map[Ref]bool)
		}
		x.extRefs[z] = true
	}
	y.extRefs = nil
}

// find returns the canonical node decl.
// (The nodes form a disjoint set forest.)
func (decl *declNode) find() *declNode {
	rep := decl.rep
	if rep != decl {
		rep = rep.find()
		decl.rep = rep // simple path compression
	}
	return rep
}

func checkCanonical(x *declNode) {
	if debug {
		assert(x == x.find(), "not canonical")
	}
}

func assert(cond bool, msg string) {
	if debug && !cond {
		panic(msg)
	}
}
