// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typerefs

import (
	"fmt"
	"go/ast"
	"go/token"
	"sort"
	"strings"

	"golang.org/x/tools/gopls/internal/astutil"
	"golang.org/x/tools/gopls/internal/lsp/frob"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/typeparams"
)

// Encode analyzes the Go syntax trees of a package, constructs a
// reference graph, and uses it to compute, for each exported
// declaration, the set of exported symbols of directly imported
// packages that it references, perhaps indirectly.
//
// It returns a serializable index of this information.
// Use Decode to expand the result.
func Encode(files []*source.ParsedGoFile, id source.PackageID, imports map[source.ImportPath]*source.Metadata) []byte {
	return index(files, id, imports)
}

// Decode decodes a serializable index of symbol
// reachability produced by Encode.
//
// Because many declarations reference the exact same set of symbols,
// the results are grouped into equivalence classes.
// Classes are sorted by Decls[0], ascending.
// The class with empty reachability is omitted.
//
// See the package documentation for more details as to what a
// reference does (and does not) represent.
func Decode(pkgIndex *PackageIndex, id source.PackageID, data []byte) []Class {
	return decode(pkgIndex, id, data)
}

// A Class is a reachability equivalence class.
//
// It attests that each exported package-level declaration in Decls
// references (perhaps indirectly) one of the external (imported)
// symbols in Refs.
//
// Because many Decls reach the same Refs,
// it is more efficient to group them into classes.
type Class struct {
	Decls []string // sorted set of names of exported decls with same reachability
	Refs  []Symbol // set of external symbols, in ascending (PackageID, Name) order
}

// A Symbol represents an external (imported) symbol
// referenced by the analyzed package.
type Symbol struct {
	Package IndexID // w.r.t. PackageIndex passed to decoder
	Name    string
}

// An IndexID is a small integer that uniquely identifies a package within a
// given PackageIndex.
type IndexID int

// -- internals --

// A symbolSet is a set of symbols used internally during index construction.
//
// TODO(adonovan): opt: evaluate unifying Symbol and symbol.
// (Encode would have to create a private PackageIndex.)
type symbolSet map[symbol]bool

// A symbol is the internal representation of an external
// (imported) symbol referenced by the analyzed package.
type symbol struct {
	pkg  source.PackageID
	name string
}

// declNode holds information about a package-level declaration
// (or more than one with the same name, in ill-typed code).
//
// It is a node in the symbol reference graph, whose outgoing edges
// are of two kinds: intRefs and extRefs.
type declNode struct {
	name string
	rep  *declNode // canonical representative of this SCC (initially self)

	// outgoing graph edges
	intRefs      map[*declNode]bool // to symbols in this package
	extRefs      symbolSet          // to imported symbols
	extRefsClass int                // extRefs equivalence class number (-1 until set at end)

	// Tarjan's SCC algorithm
	index, lowlink int32 // Tarjan numbering
	scc            int32 // -ve => on stack; 0 => unvisited; +ve => node is root of a found SCC
}

// state holds the working state of the Refs algorithm for a single package.
//
// The number of distinct symbols referenced by a single package
// (measured across all of kubernetes), was found to be:
//   - max = 1750.
//   - Several packages reference > 100 symbols.
//   - p95 = 32, p90 = 22, p50 = 8.
type state struct {
	// numbering of unique symbol sets
	class      []symbolSet    // unique symbol sets
	classIndex map[string]int // index of above (using SymbolSet.hash as key)

	// Tarjan's SCC algorithm
	index int32
	stack []*declNode
}

// getClassIndex returns the small integer (an index into
// state.class) that identifies the given set.
func (st *state) getClassIndex(set symbolSet) int {
	key := classKey(set)
	i, ok := st.classIndex[key]
	if !ok {
		i = len(st.class)
		st.classIndex[key] = i
		st.class = append(st.class, set)
	}
	return i
}

// appendSorted appends the symbols to syms, sorts by ascending
// (PackageID, name), and returns the result.
// The argument must be an empty slice, ideally with capacity len(set).
func (set symbolSet) appendSorted(syms []symbol) []symbol {
	for sym := range set {
		syms = append(syms, sym)
	}
	sort.Slice(syms, func(i, j int) bool {
		x, y := syms[i], syms[j]
		if x.pkg != y.pkg {
			return x.pkg < y.pkg
		}
		return x.name < y.name
	})
	return syms
}

// classKey returns a key such that equal keys imply equal sets.
// (e.g. a sorted string representation, or a cryptographic hash of same).
func classKey(set symbolSet) string {
	// Sort symbols into a stable order.
	// TODO(adonovan): opt: a cheap crypto hash (e.g. BLAKE2b) might
	// make a cheaper map key than a large string.
	// Try using a hasher instead of a builder.
	var s strings.Builder
	for _, sym := range set.appendSorted(make([]symbol, 0, len(set))) {
		fmt.Fprintf(&s, "%s:%s;", sym.pkg, sym.name)
	}
	return s.String()
}

// index builds the reference graph and encodes the index.
func index(pgfs []*source.ParsedGoFile, id source.PackageID, imports map[source.ImportPath]*source.Metadata) []byte {
	// First pass: gather package-level names and create a declNode for each.
	//
	// In ill-typed code, there may be multiple declarations of the
	// same name; a single declInfo node will represent them all.
	decls := make(map[string]*declNode)
	addDecl := func(id *ast.Ident) {
		if name := id.Name; name != "_" && decls[name] == nil {
			node := &declNode{name: name, extRefsClass: -1}
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
	st := &state{classIndex: make(map[string]int)}
	for _, pgf := range pgfs {
		visitFile(pgf.File, imports, decls)
	}

	// Find the strong components of the declNode graph
	// using Tarjan's algorithm, and coalesce each component.
	st.index = 1
	for _, decl := range decls {
		if decl.index == 0 { // unvisited
			st.visit(decl)
		}
	}

	// TODO(adonovan): opt: consider compressing the serialized
	// representation by recording not the classes but the DAG of
	// non-trivial union operations (the "pointer equivalence"
	// optimization of Hardekopf & Lin). Unlike that algorithm,
	// which piggybacks on SCC coalescing, in our case it would
	// be better to make a forward traversal from the exported
	// decls, since it avoids visiting unreachable nodes, and
	// results in a dense (not sparse) numbering of the sets.

	// Tabulate the unique reachability sets of
	// each exported package member.
	classNames := make(map[int][]string) // set of decls (names) for a given reachability set
	for name, decl := range decls {
		if !ast.IsExported(name) {
			continue
		}

		decl = decl.find()

		// Skip decls with empty reachability.
		if len(decl.extRefs) == 0 {
			continue
		}

		// Canonicalize the set (and memoize).
		class := decl.extRefsClass
		if class < 0 {
			class = st.getClassIndex(decl.extRefs)
			decl.extRefsClass = class
		}
		classNames[class] = append(classNames[class], name)
	}

	return encode(classNames, st.class)
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
					decl.extRefs = make(symbolSet)
				}
				decl.extRefs[symbol{depID, name}] = true
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
		// When visiting a method, there may not be a valid type declaration for
		// the receiver. In this case there is no way to refer to the method, so
		// we need not record edges.
		if from == nil {
			return
		}

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
	// Import declarations appear before all others.
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
				_, id, typeParams := astutil.UnpackRecv(d.Recv.List[0].Type)
				if id != nil {
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
				}
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

// -- strong component graph construction (plundered from go/pointer) --

// visit implements the depth-first search of Tarjan's SCC algorithm
// (see https://doi.org/10.1137/0201010).
// Precondition: x is canonical.
func (st *state) visit(x *declNode) {
	checkCanonical(x)
	x.index = st.index
	x.lowlink = st.index
	st.index++

	st.stack = append(st.stack, x) // push
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
			st.visit(y)
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
			i := len(st.stack) - 1
			y := st.stack[i]
			st.stack = st.stack[:i]

			checkCanonical(x)
			checkCanonical(y)

			if x == y {
				break // SCC is complete.
			}
			coalesce(x, y)
		}

		// Accumulate union of extRefs over
		// internal edges (to other SCCs).
		for y := range x.intRefs {
			y := y.find()
			if y == x {
				continue // already coalesced
			}
			assert(y.scc == 1, "edge to non-scc node")
			for z := range y.extRefs {
				if x.extRefs == nil {
					x.extRefs = make(symbolSet)
				}
				x.extRefs[z] = true // extRefs: x U= y
			}
		}

		x.scc = 1
	}
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
			x.extRefs = make(symbolSet)
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
		decl.rep = rep // simple path compression (no union-by-rank)
	}
	return rep
}

const debugSCC = false // enable assertions in strong-component algorithm

func checkCanonical(x *declNode) {
	if debugSCC {
		assert(x == x.find(), "not canonical")
	}
}

func assert(cond bool, msg string) {
	if debugSCC && !cond {
		panic(msg)
	}
}

// -- serialization --

// (The name says gob but in fact we use frob.)
var classesCodec = frob.CodecFor[gobClasses]()

type gobClasses struct {
	Strings []string // table of strings (PackageIDs and names)
	Classes []gobClass
}

type gobClass struct {
	Decls []int32 // indices into gobClasses.Strings
	Refs  []int32 // list of (package, name) pairs, each an index into gobClasses.Strings
}

// encode encodes the equivalence classes,
// (classNames[i], classes[i]), for i in range classes.
//
// With the current encoding, across kubernetes,
// the encoded size distribution has
// p50 = 511B, p95 = 4.4KB, max = 108K.
func encode(classNames map[int][]string, classes []symbolSet) []byte {
	payload := gobClasses{
		Classes: make([]gobClass, 0, len(classNames)),
	}

	// index of unique strings
	strings := make(map[string]int32)
	stringIndex := func(s string) int32 {
		i, ok := strings[s]
		if !ok {
			i = int32(len(payload.Strings))
			strings[s] = i
			payload.Strings = append(payload.Strings, s)
		}
		return i
	}

	var refs []symbol // recycled temporary
	for class, names := range classNames {
		set := classes[class]

		// names, sorted
		sort.Strings(names)
		gobDecls := make([]int32, len(names))
		for i, name := range names {
			gobDecls[i] = stringIndex(name)
		}

		// refs, sorted by ascending (PackageID, name)
		gobRefs := make([]int32, 0, 2*len(set))
		for _, sym := range set.appendSorted(refs[:0]) {
			gobRefs = append(gobRefs,
				stringIndex(string(sym.pkg)),
				stringIndex(sym.name))
		}
		payload.Classes = append(payload.Classes, gobClass{
			Decls: gobDecls,
			Refs:  gobRefs,
		})
	}

	return classesCodec.Encode(payload)
}

func decode(pkgIndex *PackageIndex, id source.PackageID, data []byte) []Class {
	var payload gobClasses
	classesCodec.Decode(data, &payload)

	classes := make([]Class, len(payload.Classes))
	for i, gobClass := range payload.Classes {
		decls := make([]string, len(gobClass.Decls))
		for i, decl := range gobClass.Decls {
			decls[i] = payload.Strings[decl]
		}
		refs := make([]Symbol, len(gobClass.Refs)/2)
		for i := range refs {
			pkgID := pkgIndex.IndexID(source.PackageID(payload.Strings[gobClass.Refs[2*i]]))
			name := payload.Strings[gobClass.Refs[2*i+1]]
			refs[i] = Symbol{Package: pkgID, Name: name}
		}
		classes[i] = Class{
			Decls: decls,
			Refs:  refs,
		}
	}

	// Sort by ascending Decls[0].
	// TODO(adonovan): move sort to encoder. Determinism is good.
	sort.Slice(classes, func(i, j int) bool {
		return classes[i].Decls[0] < classes[j].Decls[0]
	})

	return classes
}
