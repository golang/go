// Package lexical computes the structure of the lexical environment,
// including the definition of and references to all universal,
// package-level, file-level and function-local entities.  It does not
// record qualified identifiers, labels, struct fields, or methods.
//
// It is intended for renaming and refactoring tools, which need a more
// precise understanding of identifier resolution than is available from
// the output of the type-checker alone.
//
// THIS INTERFACE IS EXPERIMENTAL AND MAY CHANGE OR BE REMOVED IN FUTURE.
//
package lexical // import "golang.org/x/tools/refactor/lexical"

// OVERVIEW
//
// As we traverse the AST, we build a "spaghetti stack" of Blocks,
// i.e. a tree with parent edges pointing to the root.  Each time we
// visit an identifier that's a reference into the lexical environment,
// we create and save an Environment, which captures the current mapping
// state of the Block; these are saved for the client.
//
// We don't bother recording non-lexical references.

// TODO(adonovan):
// - make it robust against syntax errors.  Audit all type assertions, etc.
// - better still, after the Go 1.4 thaw, move this into go/types.
//   I don't think it need be a big change since the visitor is already there;
//   we just need to records Environments.  lexical.Block is analogous
//   to types.Scope.

import (
	"fmt"
	"go/ast"
	"go/token"
	"os"
	"strconv"

	"golang.org/x/tools/go/types"
)

const trace = false

var logf = func(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, format, args...)
}

// A Block is a level of the lexical environment, a tree of blocks.
// It maps names to objects.
//
type Block struct {
	kind   string   // one of universe package file func block if switch typeswitch case for range
	syntax ast.Node // syntax declaring the block (nil for universe and package) [needed?]

	parent   Environment
	bindings []types.Object // bindings in lexical order
	index    map[string]int // maps a name to the index of its binding, for fast lookup
}

// An Environment is a snapshot of a Block taken at a certain lexical
// position. It may contain bindings for fewer names than the
// (completed) block, or different bindings for names that are
// re-defined later in the block.
//
// For example, the lexical Block for the function f below contains a
// binding for the local var x, but the Environments captured by at the
// two print(x) calls differ: the first contains this binding, the
// second does not.  The first Environment contains a different binding
// for x: the string var defined in the package block, an ancestor.
//
//	var x string
// 	func f() {
//		print(x)
//		x := 1
//		print(x)
//	}
//
type Environment struct {
	block     *Block
	nbindings int // length of prefix of block.bindings that's visible
}

// Depth returns the depth of this block in the block tree.
// The universal block has depth 1, a package block 2, a file block 3, etc.
func (b *Block) Depth() int {
	if b == nil {
		return 0
	}
	return 1 + b.parent.block.Depth()
}

// env returns an Environment that is a snapshot of b's current state.
func (b *Block) env() Environment {
	return Environment{b, len(b.bindings)}
}

// Lookup returns the definition of name in the environment specified by
// env, and the Block that defines it, which may be an ancestor.
func (env Environment) Lookup(name string) (types.Object, *Block) {
	if env.block == nil {
		return nil, nil
	}
	return lookup(env.block, name, env.nbindings)
}

// nbindings specifies what prefix of b.bindings should be considered visible.
func lookup(b *Block, name string, nbindings int) (types.Object, *Block) {
	if b == nil {
		return nil, nil
	}
	if i, ok := b.index[name]; ok && i < nbindings {
		return b.bindings[i], b
	}

	parent := b.parent
	if parent.block == nil {
		return nil, nil
	}
	return lookup(parent.block, name, parent.nbindings)
}

// Lookup returns the definition of name in the environment specified by
// b, and the Block that defines it, which may be an ancestor.
func (b *Block) Lookup(name string) (types.Object, *Block) {
	return b.env().Lookup(name)
}

// Block returns the block of which this environment is a partial view.
func (env Environment) Block() *Block {
	return env.block
}

func (env Environment) String() string {
	return fmt.Sprintf("%s:%d", env.block, env.nbindings)
}

func (b *Block) String() string {
	var s string
	if b.parent.block != nil {
		s = b.parent.block.String()
		s += "."
	}
	return s + b.kind
}

var universe = &Block{kind: "universe", index: make(map[string]int)}

func init() {
	for i, name := range types.Universe.Names() {
		obj := types.Universe.Lookup(name)
		universe.bindings = append(universe.bindings, obj)
		universe.index[name] = i
	}
}

// -- resolver ---------------------------------------------------------

// A Reference provides the lexical environment for a given reference to
// an object in lexical scope.
type Reference struct {
	Id  *ast.Ident
	Env Environment
}

// resolver holds the state of the identifier resolution visitation:
// the package information, the result, and the current block.
type resolver struct {
	fset    *token.FileSet
	imports map[string]*types.Package
	pkg     *types.Package
	info    *types.Info

	// visitor state
	block *Block

	result *Info
}

func (r *resolver) setBlock(kind string, syntax ast.Node) *Block {
	b := &Block{
		kind:   kind,
		syntax: syntax,
		parent: r.block.env(),
		index:  make(map[string]int),
	}
	if syntax != nil {
		r.result.Blocks[syntax] = b
	}
	r.block = b
	return b
}

func (r *resolver) use(id *ast.Ident, env Environment) {
	if id.Name == "_" {
		return // an error
	}
	obj, _ := env.Lookup(id.Name)
	if obj == nil {
		logf("%s: lookup of %s failed\n", r.fset.Position(id.Pos()), id.Name)
	} else if want := r.info.Uses[id]; obj != want {
		// sanity check against go/types resolver
		logf("%s: internal error: lookup of %s yielded wrong object: got %v (%s), want %v\n",
			r.fset.Position(id.Pos()), id.Name, types.ObjectString(r.pkg, obj),
			r.fset.Position(obj.Pos()),
			want)
	}
	if trace {
		logf("use %s = %v in %s\n", id.Name, types.ObjectString(r.pkg, obj), env)
	}

	r.result.Refs[obj] = append(r.result.Refs[obj], Reference{id, env})
}

func (r *resolver) define(b *Block, id *ast.Ident) {
	obj := r.info.Defs[id]
	if obj == nil {
		logf("%s: internal error: not a defining ident: %s\n",
			r.fset.Position(id.Pos()), id.Name)
		panic(id)
	}
	r.defineObject(b, id.Name, obj)

	// Objects (other than PkgName) defined at file scope
	// are also defined in the enclosing package scope.
	if _, ok := b.syntax.(*ast.File); ok {
		switch obj.(type) {
		default:
			r.defineObject(b.parent.block, id.Name, obj)
		case nil, *types.PkgName:
		}
	}
}

// Used for implicit objects created by some ImportSpecs and CaseClauses.
func (r *resolver) defineImplicit(b *Block, n ast.Node, name string) {
	obj := r.info.Implicits[n]
	if obj == nil {
		logf("%s: internal error: not an implicit definition: %T\n",
			r.fset.Position(n.Pos()), n)
	}
	r.defineObject(b, name, obj)
}

func (r *resolver) defineObject(b *Block, name string, obj types.Object) {
	if obj.Name() == "_" {
		return
	}
	i := len(b.bindings)
	b.bindings = append(b.bindings, obj)
	b.index[name] = i
	if trace {
		logf("def %s = %s in %s\n", name, types.ObjectString(r.pkg, obj), b)
	}
	r.result.Defs[obj] = b
}

func (r *resolver) function(recv *ast.FieldList, typ *ast.FuncType, body *ast.BlockStmt, syntax ast.Node) {
	// Use all signature types in enclosing block.
	r.expr(typ)
	r.fieldList(recv, false)

	savedBlock := r.block // save
	r.setBlock("func", syntax)

	// Define all parameters/results, and visit the body, within the func block.
	r.fieldList(typ.Params, true)
	r.fieldList(typ.Results, true)
	r.fieldList(recv, true)
	if body != nil {
		r.stmtList(body.List)
	}

	r.block = savedBlock // restore
}

func (r *resolver) fieldList(list *ast.FieldList, def bool) {
	if list != nil {
		for _, f := range list.List {
			if def {
				for _, id := range f.Names {
					r.define(r.block, id)
				}
			} else {
				r.expr(f.Type)
			}
		}
	}
}

func (r *resolver) exprList(list []ast.Expr) {
	for _, x := range list {
		r.expr(x)
	}
}

func (r *resolver) expr(n ast.Expr) {
	switch n := n.(type) {
	case *ast.BadExpr:
	case *ast.BasicLit:
		// no-op

	case *ast.Ident:
		r.use(n, r.block.env())

	case *ast.Ellipsis:
		if n.Elt != nil {
			r.expr(n.Elt)
		}

	case *ast.FuncLit:
		r.function(nil, n.Type, n.Body, n)

	case *ast.CompositeLit:
		if n.Type != nil {
			r.expr(n.Type)
		}
		tv := r.info.Types[n]
		if _, ok := deref(tv.Type).Underlying().(*types.Struct); ok {
			for _, elt := range n.Elts {
				if kv, ok := elt.(*ast.KeyValueExpr); ok {
					r.expr(kv.Value)

					// Also uses field kv.Key (non-lexical)
					//  id := kv.Key.(*ast.Ident)
					//  obj := r.info.Uses[id]
					//  logf("use %s = %v (field)\n",
					// 	id.Name, types.ObjectString(r.pkg, obj))
					// TODO make a fake FieldVal selection?
				} else {
					r.expr(elt)
				}
			}
		} else {
			r.exprList(n.Elts)
		}

	case *ast.ParenExpr:
		r.expr(n.X)

	case *ast.SelectorExpr:
		r.expr(n.X)

		// Non-lexical reference to field/method, or qualified identifier.
		// if sel, ok := r.info.Selections[n]; ok { // selection
		// 	switch sel.Kind() {
		// 	case types.FieldVal:
		// 		logf("use %s = %v (field)\n",
		// 			n.Sel.Name, types.ObjectString(r.pkg, sel.Obj()))
		// 	case types.MethodExpr, types.MethodVal:
		// 		logf("use %s = %v (method)\n",
		// 			n.Sel.Name, types.ObjectString(r.pkg, sel.Obj()))
		// 	}
		// } else { // qualified identifier
		// 	obj := r.info.Uses[n.Sel]
		// 	logf("use %s = %v (qualified)\n", n.Sel.Name, obj)
		// }

	case *ast.IndexExpr:
		r.expr(n.X)
		r.expr(n.Index)

	case *ast.SliceExpr:
		r.expr(n.X)
		if n.Low != nil {
			r.expr(n.Low)
		}
		if n.High != nil {
			r.expr(n.High)
		}
		if n.Max != nil {
			r.expr(n.Max)
		}

	case *ast.TypeAssertExpr:
		r.expr(n.X)
		if n.Type != nil {
			r.expr(n.Type)
		}

	case *ast.CallExpr:
		r.expr(n.Fun)
		r.exprList(n.Args)

	case *ast.StarExpr:
		r.expr(n.X)

	case *ast.UnaryExpr:
		r.expr(n.X)

	case *ast.BinaryExpr:
		r.expr(n.X)
		r.expr(n.Y)

	case *ast.KeyValueExpr:
		r.expr(n.Key)
		r.expr(n.Value)

	case *ast.ArrayType:
		if n.Len != nil {
			r.expr(n.Len)
		}
		r.expr(n.Elt)

	case *ast.StructType:
		// Use all the type names, but don't define any fields.
		r.fieldList(n.Fields, false)

	case *ast.FuncType:
		// Use all the type names, but don't define any vars.
		r.fieldList(n.Params, false)
		r.fieldList(n.Results, false)

	case *ast.InterfaceType:
		// Use all the type names, but don't define any methods.
		r.fieldList(n.Methods, false)

	case *ast.MapType:
		r.expr(n.Key)
		r.expr(n.Value)

	case *ast.ChanType:
		r.expr(n.Value)

	default:
		panic(n)
	}
}

func (r *resolver) stmtList(list []ast.Stmt) {
	for _, s := range list {
		r.stmt(s)
	}
}

func (r *resolver) stmt(n ast.Stmt) {
	switch n := n.(type) {
	case *ast.BadStmt:
	case *ast.EmptyStmt:
		// nothing to do

	case *ast.DeclStmt:
		decl := n.Decl.(*ast.GenDecl)
		for _, spec := range decl.Specs {
			switch spec := spec.(type) {
			case *ast.ValueSpec: // const or var
				if spec.Type != nil {
					r.expr(spec.Type)
				}
				r.exprList(spec.Values)
				for _, name := range spec.Names {
					r.define(r.block, name)
				}

			case *ast.TypeSpec:
				r.define(r.block, spec.Name)
				r.expr(spec.Type)
			}
		}

	case *ast.LabeledStmt:
		// Also defines label n.Label (non-lexical)
		r.stmt(n.Stmt)

	case *ast.ExprStmt:
		r.expr(n.X)

	case *ast.SendStmt:
		r.expr(n.Chan)
		r.expr(n.Value)

	case *ast.IncDecStmt:
		r.expr(n.X)

	case *ast.AssignStmt:
		if n.Tok == token.DEFINE {
			r.exprList(n.Rhs)
			for _, lhs := range n.Lhs {
				id := lhs.(*ast.Ident)
				if _, ok := r.info.Defs[id]; ok {
					r.define(r.block, id)
				} else {
					r.use(id, r.block.env())
				}
			}
		} else { // ASSIGN
			r.exprList(n.Lhs)
			r.exprList(n.Rhs)
		}

	case *ast.GoStmt:
		r.expr(n.Call)

	case *ast.DeferStmt:
		r.expr(n.Call)

	case *ast.ReturnStmt:
		r.exprList(n.Results)

	case *ast.BranchStmt:
		if n.Label != nil {
			// Also uses label n.Label (non-lexical)
		}

	case *ast.SelectStmt:
		r.stmtList(n.Body.List)

	case *ast.BlockStmt: // (explicit blocks only)
		savedBlock := r.block // save
		r.setBlock("block", n)
		r.stmtList(n.List)
		r.block = savedBlock // restore

	case *ast.IfStmt:
		savedBlock := r.block // save
		r.setBlock("if", n)
		if n.Init != nil {
			r.stmt(n.Init)
		}
		r.expr(n.Cond)
		r.stmt(n.Body) // new block
		if n.Else != nil {
			r.stmt(n.Else)
		}
		r.block = savedBlock // restore

	case *ast.CaseClause:
		savedBlock := r.block // save
		r.setBlock("case", n)
		if obj, ok := r.info.Implicits[n]; ok {
			// e.g.
			//   switch y := x.(type) {
			//   case T: // we declare an implicit 'var y T' in this block
			//   }
			r.defineImplicit(r.block, n, obj.Name())
		}
		r.exprList(n.List)
		r.stmtList(n.Body)
		r.block = savedBlock // restore

	case *ast.SwitchStmt:
		savedBlock := r.block // save
		r.setBlock("switch", n)
		if n.Init != nil {
			r.stmt(n.Init)
		}
		if n.Tag != nil {
			r.expr(n.Tag)
		}
		r.stmtList(n.Body.List)
		r.block = savedBlock // restore

	case *ast.TypeSwitchStmt:
		savedBlock := r.block // save
		r.setBlock("typeswitch", n)
		if n.Init != nil {
			r.stmt(n.Init)
		}
		if assign, ok := n.Assign.(*ast.AssignStmt); ok { // y := x.(type)
			r.expr(assign.Rhs[0]) // skip y: not a defining ident
		} else {
			r.stmt(n.Assign)
		}
		r.stmtList(n.Body.List)
		r.block = savedBlock // restore

	case *ast.CommClause:
		savedBlock := r.block // save
		r.setBlock("case", n)
		if n.Comm != nil {
			r.stmt(n.Comm)
		}
		r.stmtList(n.Body)
		r.block = savedBlock // restore

	case *ast.ForStmt:
		savedBlock := r.block // save
		r.setBlock("for", n)
		if n.Init != nil {
			r.stmt(n.Init)
		}
		if n.Cond != nil {
			r.expr(n.Cond)
		}
		if n.Post != nil {
			r.stmt(n.Post)
		}
		r.stmt(n.Body)
		r.block = savedBlock // restore

	case *ast.RangeStmt:
		r.expr(n.X)
		savedBlock := r.block // save
		r.setBlock("range", n)
		if n.Tok == token.DEFINE {
			if n.Key != nil {
				r.define(r.block, n.Key.(*ast.Ident))
			}
			if n.Value != nil {
				r.define(r.block, n.Value.(*ast.Ident))
			}
		} else {
			if n.Key != nil {
				r.expr(n.Key)
			}
			if n.Value != nil {
				r.expr(n.Value)
			}
		}
		r.stmt(n.Body)
		r.block = savedBlock // restore

	default:
		panic(n)
	}
}

func (r *resolver) doImport(s *ast.ImportSpec, fileBlock *Block) {
	path, _ := strconv.Unquote(s.Path.Value)
	pkg := r.imports[path]
	if s.Name == nil { // normal
		r.defineImplicit(fileBlock, s, pkg.Name())
	} else if s.Name.Name == "." { // dot import
		for _, name := range pkg.Scope().Names() {
			if ast.IsExported(name) {
				obj := pkg.Scope().Lookup(name)
				r.defineObject(fileBlock, name, obj)
			}
		}
	} else { // renaming import
		r.define(fileBlock, s.Name)
	}
}

func (r *resolver) doPackage(pkg *types.Package, files []*ast.File) {
	r.block = universe
	r.result.Blocks[nil] = universe

	r.result.PackageBlock = r.setBlock("package", nil)

	var fileBlocks []*Block

	// 1. Insert all package-level objects into file and package blocks.
	//    (PkgName objects are only inserted into file blocks.)
	for _, f := range files {
		r.block = r.result.PackageBlock
		fileBlock := r.setBlock("file", f) // package is not yet visible to file
		fileBlocks = append(fileBlocks, fileBlock)

		for _, d := range f.Decls {
			switch d := d.(type) {
			case *ast.GenDecl:
				for _, s := range d.Specs {
					switch s := s.(type) {
					case *ast.ImportSpec:
						r.doImport(s, fileBlock)

					case *ast.ValueSpec: // const or var
						for _, name := range s.Names {
							r.define(r.result.PackageBlock, name)
						}

					case *ast.TypeSpec:
						r.define(r.result.PackageBlock, s.Name)
					}
				}

			case *ast.FuncDecl:
				if d.Recv == nil { // function
					if d.Name.Name != "init" {
						r.define(r.result.PackageBlock, d.Name)
					}
				}
			}
		}
	}

	// 2. Now resolve bodies of GenDecls and FuncDecls.
	for i, f := range files {
		fileBlock := fileBlocks[i]
		fileBlock.parent = r.result.PackageBlock.env() // make entire package visible to this file

		for _, d := range f.Decls {
			r.block = fileBlock

			switch d := d.(type) {
			case *ast.GenDecl:
				for _, s := range d.Specs {
					switch s := s.(type) {
					case *ast.ValueSpec: // const or var
						if s.Type != nil {
							r.expr(s.Type)
						}
						r.exprList(s.Values)

					case *ast.TypeSpec:
						r.expr(s.Type)
					}
				}

			case *ast.FuncDecl:
				r.function(d.Recv, d.Type, d.Body, d)
			}
		}
	}

	r.block = nil
}

// An Info contains the lexical reference structure of a package.
type Info struct {
	Defs         map[types.Object]*Block      // maps each object to its defining lexical block
	Refs         map[types.Object][]Reference // maps each object to the set of references to it
	Blocks       map[ast.Node]*Block          // maps declaring syntax to block; nil => universe
	PackageBlock *Block                       // the package-level lexical block
}

// Structure computes the structure of the lexical environment of the
// package specified by (pkg, info, files).
//
// The info.{Types,Defs,Uses,Implicits} maps must have been populated
// by the type-checker
//
// fset is used for logging.
//
func Structure(fset *token.FileSet, pkg *types.Package, info *types.Info, files []*ast.File) *Info {
	r := resolver{
		fset:    fset,
		imports: make(map[string]*types.Package),
		result: &Info{
			Defs:   make(map[types.Object]*Block),
			Refs:   make(map[types.Object][]Reference),
			Blocks: make(map[ast.Node]*Block),
		},
		pkg:  pkg,
		info: info,
	}

	// Build import map for just this package.
	r.imports["unsafe"] = types.Unsafe
	for _, imp := range pkg.Imports() {
		r.imports[imp.Path()] = imp
	}

	r.doPackage(pkg, files)

	return r.result
}

// -- Plundered from golang.org/x/tools/go/ssa -----------------

// deref returns a pointer's element type; otherwise it returns typ.
func deref(typ types.Type) types.Type {
	if p, ok := typ.Underlying().(*types.Pointer); ok {
		return p.Elem()
	}
	return typ
}
