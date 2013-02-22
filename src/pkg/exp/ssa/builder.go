package ssa

// This file defines the SSA builder.
//
// The builder has two phases, CREATE and BUILD.  In the CREATE
// phase, all packages are constructed and type-checked and
// definitions of all package members are created, method-sets are
// computed, and bridge methods are synthesized.  The create phase
// proceeds in topological order over the import dependency graph,
// initiated by client calls to CreatePackage.
//
// In the BUILD phase, the Builder traverses the AST of each Go source
// function and generates SSA instructions for the function body.
// Within each package, building proceeds in a topological order over
// the symbol reference graph, whose roots are the set of
// package-level declarations in lexical order.
//
// In principle, the BUILD phases for each package can occur in
// parallel, and that is our goal though there remains work to do.
// Currently we ensure that all the imports of a package are fully
// built before we start building it.
//
// The Builder's and Program's indices (maps) are populated and
// mutated during the CREATE phase, but during the BUILD phase they
// remain constant, with the following exceptions:
// - demoteSelector mutates Builder.types during the BUILD phase.
//   TODO(adonovan): fix: let's not do that.
// - globalValueSpec mutates Builder.nTo1Vars.
//   TODO(adonovan): make this a per-Package map so it's thread-safe.
// - Program.methodSets is populated lazily across phases.
//   It uses a mutex so that access from multiple threads is serialized.

// TODO(adonovan): fix the following:
// - support f(g()) where g has multiple result parameters.
// - concurrent SSA code generation of multiple packages.

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"os"
	"strconv"
)

var (
	varOk = &types.Var{Name: "ok", Type: tBool}

	// Type constants.
	tBool       = types.Typ[types.Bool]
	tByte       = types.Typ[types.Byte]
	tFloat32    = types.Typ[types.Float32]
	tFloat64    = types.Typ[types.Float64]
	tComplex64  = types.Typ[types.Complex64]
	tComplex128 = types.Typ[types.Complex128]
	tInt        = types.Typ[types.Int]
	tInvalid    = types.Typ[types.Invalid]
	tUntypedNil = types.Typ[types.UntypedNil]
	tRangeIter  = &types.Basic{Name: "iter"} // the type of all "range" iterators
	tEface      = new(types.Interface)

	// The result type of a "select".
	tSelect = &types.Result{Values: []*types.Var{
		{Name: "index", Type: tInt},
		{Name: "recv", Type: tInvalid},
		varOk,
	}}

	// SSA Value constants.
	vZero  = intLiteral(0)
	vOne   = intLiteral(1)
	vTrue  = newLiteral(true, tBool)
	vFalse = newLiteral(false, tBool)
)

// A Builder creates the SSA representation of a single program.
// Instances may be created using NewBuilder.
//
// The SSA Builder constructs a Program containing Package instances
// for packages of Go source code, loading, parsing and recursively
// constructing packages for all imported dependencies as well.
//
// If the UseGCImporter mode flag is specified, binary object files
// produced by the gc compiler will be loaded instead of source code
// for all imported packages.  Such files supply only the types of
// package-level declarations and values of constants, but no code, so
// this mode will not yield a whole program.  It is intended for
// analyses that perform intraprocedural analysis of a single package.
//
// A typical client will create a Builder with NewBuilder; call
// CreatePackage for the "root" package(s), e.g. main; then call
// BuildPackage on the same set of packages to construct SSA-form code
// for functions and methods.  After that, the representation of the
// program (Builder.Prog) is complete and transitively closed, and the
// Builder object can be discarded to reclaim its memory.  The
// client's analysis may then begin.
//
type Builder struct {
	Prog        *Program                    // the program being built
	mode        BuilderMode                 // set of mode bits
	loader      SourceLoader                // the loader for imported source files
	importErrs  map[string]error            // across-packages import cache of failures
	packages    map[*types.Package]*Package // SSA packages by types.Package
	types       map[ast.Expr]types.Type     // inferred types of expressions
	constants   map[ast.Expr]*Literal       // values of constant expressions
	idents      map[*ast.Ident]types.Object // canonical type objects of all named entities
	globals     map[types.Object]Value      // all package-level funcs and vars, and universal built-ins
	nTo1Vars    map[*ast.ValueSpec]bool     // set of n:1 ValueSpecs already built [not threadsafe]
	typechecker types.Context               // the typechecker context (stateless)
}

// BuilderMode is a bitmask of options for diagnostics and checking.
type BuilderMode uint

const (
	LogPackages          BuilderMode = 1 << iota // Dump package inventory to stderr
	LogFunctions                                 // Dump function SSA code to stderr
	LogSource                                    // Show source locations as SSA builder progresses
	SanityCheckFunctions                         // Perform sanity checking of function bodies
	UseGCImporter                                // Ignore SourceLoader; use gc-compiled object code for all imports
	NaiveForm                                    // Build naïve SSA form: don't replace local loads/stores with registers
)

// NewBuilder creates and returns a new SSA builder.
//
// mode is a bitfield of options controlling verbosity, logging and
// additional sanity checks.
//
// loader is a SourceLoader function that finds, loads and parses Go
// source files for a given import path.  (It is ignored if the mode
// bits include UseGCImporter.)
//
// errh is an optional error handler that is called for each error
// encountered during type checking; if nil, only the first type error
// will be returned, via the result of CreatePackage.
//
func NewBuilder(mode BuilderMode, loader SourceLoader, errh func(error)) *Builder {
	b := &Builder{
		Prog: &Program{
			Files:           token.NewFileSet(),
			Packages:        make(map[string]*Package),
			Builtins:        make(map[types.Object]*Builtin),
			methodSets:      make(map[types.Type]MethodSet),
			concreteMethods: make(map[*types.Method]*Function),
			mode:            mode,
		},
		mode:       mode,
		loader:     loader,
		constants:  make(map[ast.Expr]*Literal),
		globals:    make(map[types.Object]Value),
		idents:     make(map[*ast.Ident]types.Object),
		importErrs: make(map[string]error),
		nTo1Vars:   make(map[*ast.ValueSpec]bool),
		packages:   make(map[*types.Package]*Package),
		types:      make(map[ast.Expr]types.Type),
	}

	// TODO(adonovan): opt: record the Expr/Ident calls into
	// constants/idents/types maps associated with the containing
	// package so we can discard them once that package is built.
	b.typechecker = types.Context{
		Error: errh,
		Expr: func(x ast.Expr, typ types.Type, val interface{}) {
			b.types[x] = typ
			if val != nil {
				b.constants[x] = newLiteral(val, typ)
			}
		},
		Ident: func(ident *ast.Ident, obj types.Object) {
			// Invariants:
			// - obj is non-nil.
			// - isBlankIdent(ident) <=> obj.GetType()==nil
			b.idents[ident] = obj
		},
		Import: func(imports map[string]*types.Package, path string) (pkg *types.Package, err error) {
			return b.doImport(imports, path)
		},
	}

	// Create Values for built-in functions.
	for _, obj := range types.Universe.Entries {
		switch obj := obj.(type) {
		case *types.Func:
			v := &Builtin{obj}
			b.globals[obj] = v
			b.Prog.Builtins[obj] = v
		}
	}
	return b
}

// isPackageRef returns the identity of the object if sel is a
// package-qualified reference to a named const, var, func or type.
// Otherwise it returns nil.
//
// It's unfortunate that this is a method of builder, but even the
// small amount of name resolution required here is no longer
// available on the AST.
//
func (b *Builder) isPackageRef(sel *ast.SelectorExpr) types.Object {
	if id, ok := sel.X.(*ast.Ident); ok {
		if obj := b.obj(id); objKind(obj) == ast.Pkg {
			return obj.(*types.Package).Scope.Lookup(sel.Sel.Name)
		}
	}
	return nil
}

// isType returns true iff expression e denotes a type.
//
// It's unfortunate that this is a method of builder, but even the
// small amount of name resolution required here is no longer
// available on the AST.
//
func (b *Builder) isType(e ast.Expr) bool {
	switch e := e.(type) {
	case *ast.SelectorExpr: // pkg.Type
		if obj := b.isPackageRef(e); obj != nil {
			return objKind(obj) == ast.Typ
		}
	case *ast.StarExpr: // *T
		return b.isType(e.X)
	case *ast.Ident:
		return objKind(b.obj(e)) == ast.Typ
	case *ast.ArrayType, *ast.StructType, *ast.FuncType, *ast.InterfaceType, *ast.MapType, *ast.ChanType:
		return true
	case *ast.ParenExpr:
		return b.isType(e.X)
	}
	return false
}

// lookup returns the package-level *Function or *Global (or universal
// *Builtin) for the named object obj, causing its initialization code
// to be emitted into v.Package.Init if not already done.
//
func (b *Builder) lookup(obj types.Object) (v Value, ok bool) {
	v, ok = b.globals[obj]
	if ok {
		// TODO(adonovan): the build phase should only
		// propagate to v if it's in the same package as the
		// caller of lookup if we want to make this
		// concurrent.
		switch v := v.(type) {
		case *Function:
			b.buildFunction(v)
		case *Global:
			b.buildGlobal(v, obj)
		}
	}
	return
}

// exprType(e) returns the type of expression e.
// Callers should not access b.types directly since it may lie if
// called from within a typeswitch.
//
func (b *Builder) exprType(e ast.Expr) types.Type {
	// For Ident, b.types may be more specific than
	// b.obj(id.(*ast.Ident)).GetType(),
	// e.g. in the case of typeswitch.
	if t, ok := b.types[e]; ok {
		return t
	}
	// The typechecker doesn't notify us of all Idents,
	// e.g. s.Key and s.Value in a RangeStmt.
	// So we have this fallback.
	// TODO(gri): Is this a typechecker bug?  If so, eliminate
	// this case and panic.
	if id, ok := e.(*ast.Ident); ok {
		return b.obj(id).GetType()
	}
	panic("no type for expression")
}

// obj returns the typechecker object denoted by the specified
// identifier.  Panic ensues if there is none.
//
func (b *Builder) obj(id *ast.Ident) types.Object {
	if obj, ok := b.idents[id]; ok {
		return obj
	}
	panic(fmt.Sprintf("no types.Object for ast.Ident %s @ %p", id.Name, id))
}

// cond emits to fn code to evaluate boolean condition e and jump
// to t or f depending on its value, performing various simplifications.
//
// Postcondition: fn.currentBlock is nil.
//
func (b *Builder) cond(fn *Function, e ast.Expr, t, f *BasicBlock) {
	switch e := e.(type) {
	case *ast.ParenExpr:
		b.cond(fn, e.X, t, f)
		return

	case *ast.BinaryExpr:
		switch e.Op {
		case token.LAND:
			ltrue := fn.newBasicBlock("cond.true")
			b.cond(fn, e.X, ltrue, f)
			fn.currentBlock = ltrue
			b.cond(fn, e.Y, t, f)
			return

		case token.LOR:
			lfalse := fn.newBasicBlock("cond.false")
			b.cond(fn, e.X, t, lfalse)
			fn.currentBlock = lfalse
			b.cond(fn, e.Y, t, f)
			return
		}

	case *ast.UnaryExpr:
		if e.Op == token.NOT {
			b.cond(fn, e.X, f, t)
			return
		}
	}

	switch cond := b.expr(fn, e).(type) {
	case *Literal:
		// Dispatch constant conditions statically.
		if cond.Value.(bool) {
			emitJump(fn, t)
		} else {
			emitJump(fn, f)
		}
	default:
		emitIf(fn, cond, t, f)
	}
}

// logicalBinop emits code to fn to evaluate e, a &&- or
// ||-expression whose reified boolean value is wanted.
// The value is returned.
//
func (b *Builder) logicalBinop(fn *Function, e *ast.BinaryExpr) Value {
	rhs := fn.newBasicBlock("binop.rhs")
	done := fn.newBasicBlock("binop.done")

	var short Value // value of the short-circuit path
	switch e.Op {
	case token.LAND:
		b.cond(fn, e.X, rhs, done)
		short = vFalse
	case token.LOR:
		b.cond(fn, e.X, done, rhs)
		short = vTrue
	}

	// Is rhs unreachable?
	if rhs.Preds == nil {
		// Simplify false&&y to false, true||y to true.
		fn.currentBlock = done
		return short
	}

	// Is done unreachable?
	if done.Preds == nil {
		// Simplify true&&y (or false||y) to y.
		fn.currentBlock = rhs
		return b.expr(fn, e.Y)
	}

	// All edges from e.X to done carry the short-circuit value.
	var edges []Value
	for _ = range done.Preds {
		edges = append(edges, short)
	}

	// The edge from e.Y to done carries the value of e.Y.
	fn.currentBlock = rhs
	edges = append(edges, b.expr(fn, e.Y))
	emitJump(fn, done)
	fn.currentBlock = done

	// TODO(adonovan): do we need emitConv on each edge?
	// Test with named boolean types.
	phi := &Phi{Edges: edges, Comment: e.Op.String()}
	phi.Type_ = phi.Edges[0].Type()
	return done.emit(phi)
}

// exprN lowers a multi-result expression e to SSA form, emitting code
// to fn and returning a single Value whose type is a *types.Results
// (tuple).  The caller must access the components via Extract.
//
// Multi-result expressions include CallExprs in a multi-value
// assignment or return statement, and "value,ok" uses of
// TypeAssertExpr, IndexExpr (when X is a map), and UnaryExpr (when Op
// is token.ARROW).
//
func (b *Builder) exprN(fn *Function, e ast.Expr) Value {
	var typ types.Type
	var tuple Value
	switch e := e.(type) {
	case *ast.ParenExpr:
		return b.exprN(fn, e.X)

	case *ast.CallExpr:
		// Currently, no built-in function nor type conversion
		// has multiple results, so we can avoid some of the
		// cases for single-valued CallExpr.
		var c Call
		b.setCall(fn, e, &c.CallCommon)
		c.Type_ = b.exprType(e)
		return fn.emit(&c)

	case *ast.IndexExpr:
		mapt := underlyingType(b.exprType(e.X)).(*types.Map)
		typ = mapt.Elt
		tuple = fn.emit(&Lookup{
			X:       b.expr(fn, e.X),
			Index:   emitConv(fn, b.expr(fn, e.Index), mapt.Key),
			CommaOk: true,
		})

	case *ast.TypeAssertExpr:
		typ = b.exprType(e)
		tuple = fn.emit(&TypeAssert{
			X:            b.expr(fn, e.X),
			AssertedType: typ,
			CommaOk:      true,
		})

	case *ast.UnaryExpr: // must be receive <-
		typ = underlyingType(b.exprType(e.X)).(*types.Chan).Elt
		tuple = fn.emit(&UnOp{
			Op:      token.ARROW,
			X:       b.expr(fn, e.X),
			CommaOk: true,
		})

	default:
		panic(fmt.Sprintf("unexpected exprN: %T", e))
	}

	// The typechecker sets the type of the expression to just the
	// asserted type in the "value, ok" form, not to *types.Result
	// (though it includes the valueOk operand in its error messages).

	tuple.(interface {
		setType(types.Type)
	}).setType(&types.Result{Values: []*types.Var{
		{Name: "value", Type: typ},
		varOk,
	}})
	return tuple
}

// builtin emits to fn SSA instructions to implement a call to the
// built-in function called name with the specified arguments
// and return type.  It returns the value defined by the result.
//
// The result is nil if no special handling was required; in this case
// the caller should treat this like an ordinary library function
// call.
//
func (b *Builder) builtin(fn *Function, name string, args []ast.Expr, typ types.Type) Value {
	switch name {
	case "make":
		switch underlyingType(typ).(type) {
		case *types.Slice:
			n := b.expr(fn, args[1])
			m := n
			if len(args) == 3 {
				m = b.expr(fn, args[2])
			}
			v := &MakeSlice{
				Len: n,
				Cap: m,
			}
			v.setType(typ)
			return fn.emit(v)

		case *types.Map:
			var res Value
			if len(args) == 2 {
				res = b.expr(fn, args[1])
			}
			v := &MakeMap{Reserve: res}
			v.setType(typ)
			return fn.emit(v)

		case *types.Chan:
			var sz Value = vZero
			if len(args) == 2 {
				sz = b.expr(fn, args[1])
			}
			v := &MakeChan{Size: sz}
			v.setType(typ)
			return fn.emit(v)
		}

	case "new":
		return emitNew(fn, indirectType(underlyingType(typ)))

	case "len", "cap":
		// Special case: len or cap of an array or *array is
		// based on the type, not the value which may be nil.
		// We must still evaluate the value, though.  (If it
		// was side-effect free, the whole call would have
		// been constant-folded.)
		t := underlyingType(deref(b.exprType(args[0])))
		if at, ok := t.(*types.Array); ok {
			b.expr(fn, args[0]) // for effects only
			return intLiteral(at.Len)
		}
		// Otherwise treat as normal.

	case "panic":
		fn.emit(&Panic{X: emitConv(fn, b.expr(fn, args[0]), tEface)})
		fn.currentBlock = fn.newBasicBlock("unreachable")
		return vFalse // any non-nil Value will do
	}
	return nil // treat all others as a regular function call
}

// demoteSelector returns a SelectorExpr syntax tree that is
// equivalent to sel but contains no selections of promoted fields.
// It returns the field index of the explicit (=outermost) selection.
//
// pkg is the package in which the reference occurs.  This is
// significant because a non-exported field is considered distinct
// from a field of that name in any other package.
//
// This is a rather clunky and inefficient implementation, but it
// (a) is simple and hopefully self-evidently correct and
// (b) permits us to decouple the demotion from the code generation,
// the latter being performed in two modes: addr() for lvalues,
// expr() for rvalues.
// It does require mutation of Builder.types though; if we want to
// make the Builder concurrent we'll have to avoid that.
// TODO(adonovan): emit code directly rather than desugaring the AST.
//
func (b *Builder) demoteSelector(sel *ast.SelectorExpr, pkg *Package) (sel2 *ast.SelectorExpr, index int) {
	id := makeId(sel.Sel.Name, pkg.Types)
	xtype := b.exprType(sel.X)
	// fmt.Fprintln(os.Stderr, xtype, id) // debugging
	st := underlyingType(deref(xtype)).(*types.Struct)
	for i, f := range st.Fields {
		if IdFromQualifiedName(f.QualifiedName) == id {
			return sel, i
		}
	}
	// Not a named field.  Use breadth-first algorithm.
	path, index := findPromotedField(st, id)
	if path == nil {
		panic("field not found, even with promotion: " + sel.Sel.Name)
	}

	// makeSelector(e, [C,B,A]) returns (((e.A).B).C).
	// e is the original selector's base.
	// This function has no free variables.
	var makeSelector func(b *Builder, e ast.Expr, path *anonFieldPath) *ast.SelectorExpr
	makeSelector = func(b *Builder, e ast.Expr, path *anonFieldPath) *ast.SelectorExpr {
		x := e
		if path.tail != nil {
			x = makeSelector(b, e, path.tail)
		}
		sel := &ast.SelectorExpr{
			X:   x,
			Sel: &ast.Ident{Name: path.field.Name},
		}
		b.types[sel] = path.field.Type // TODO(adonovan): fix: not thread-safe
		return sel
	}

	// Construct new SelectorExpr, bottom up.
	sel2 = &ast.SelectorExpr{
		X:   makeSelector(b, sel.X, path),
		Sel: sel.Sel,
	}
	b.types[sel2] = b.exprType(sel) // TODO(adonovan): fix: not thread-safe
	return
}

// addr lowers a single-result addressable expression e to SSA form,
// emitting code to fn and returning the location (an lvalue) defined
// by the expression.
//
// If escaping is true, addr marks the base variable of the
// addressable expression e as being a potentially escaping pointer
// value.  For example, in this code:
//
//   a := A{
//     b: [1]B{B{c: 1}}
//   }
//   return &a.b[0].c
//
// the application of & causes a.b[0].c to have its address taken,
// which means that ultimately the local variable a must be
// heap-allocated.  This is a simple but very conservative escape
// analysis.
//
// Operations forming potentially escaping pointers include:
// - &x
// - a[:] iff a is an array (not *array)
// - references to variables in lexically enclosing functions.
//
func (b *Builder) addr(fn *Function, e ast.Expr, escaping bool) lvalue {
	switch e := e.(type) {
	case *ast.Ident:
		obj := b.obj(e)
		v, ok := b.lookup(obj) // var (address)
		if !ok {
			v = fn.lookup(obj, escaping)
		}
		return address{v}

	case *ast.CompositeLit:
		t := deref(b.exprType(e))
		var v Value
		if escaping {
			v = emitNew(fn, t)
		} else {
			v = fn.addLocal(t)
		}
		b.compLit(fn, v, e, t) // initialize in place
		return address{v}

	case *ast.ParenExpr:
		return b.addr(fn, e.X, escaping)

	case *ast.SelectorExpr:
		// p.M where p is a package.
		if obj := b.isPackageRef(e); obj != nil {
			if v, ok := b.lookup(obj); ok {
				return address{v}
			}
			panic("undefined package-qualified name: " + obj.GetName())
		}

		// e.f where e is an expression.
		e, index := b.demoteSelector(e, fn.Pkg)
		var x Value
		switch underlyingType(b.exprType(e.X)).(type) {
		case *types.Struct:
			x = b.addr(fn, e.X, escaping).(address).addr
		case *types.Pointer:
			x = b.expr(fn, e.X)
		}
		v := &FieldAddr{
			X:     x,
			Field: index,
		}
		v.setType(pointer(b.exprType(e)))
		return address{fn.emit(v)}

	case *ast.IndexExpr:
		var x Value
		var et types.Type
		switch t := underlyingType(b.exprType(e.X)).(type) {
		case *types.Array:
			x = b.addr(fn, e.X, escaping).(address).addr
			et = pointer(t.Elt)
		case *types.Pointer: // *array
			x = b.expr(fn, e.X)
			et = pointer(underlyingType(t.Base).(*types.Array).Elt)
		case *types.Slice:
			x = b.expr(fn, e.X)
			et = pointer(t.Elt)
		case *types.Map:
			return &element{
				m: b.expr(fn, e.X),
				k: emitConv(fn, b.expr(fn, e.Index), t.Key),
				t: t.Elt,
			}
		default:
			panic("unexpected container type in IndexExpr: " + t.String())
		}
		v := &IndexAddr{
			X:     x,
			Index: emitConv(fn, b.expr(fn, e.Index), tInt),
		}
		v.setType(et)
		return address{fn.emit(v)}

	case *ast.StarExpr:
		return address{b.expr(fn, e.X)}
	}

	panic(fmt.Sprintf("unexpected address expression: %T", e))
}

// exprInPlace emits to fn code to initialize the lvalue loc with the
// value of expression e.
//
// typ is the type of the lvalue, which may be provided by the caller
// since it is sometimes only an inherited attribute (e.g. within in
// composite literals).
//
// This is equivalent to loc.store(fn, b.expr(fn, e)) but may
// generate better code in some cases, e.g. for composite literals
// in an addressable location.
//
func (b *Builder) exprInPlace(fn *Function, loc lvalue, e ast.Expr) {
	if addr, ok := loc.(address); ok {
		if e, ok := e.(*ast.CompositeLit); ok {
			typ := addr.typ()
			switch underlyingType(typ).(type) {
			case *types.Pointer: // implicit & -- possibly escaping
				ptr := b.addr(fn, e, true).(address).addr
				addr.store(fn, ptr) // copy address
				return

			case *types.Interface:
				// e.g. var x interface{} = T{...}
				// Can't in-place initialize an interface value.
				// Fall back to copying.

			default:
				b.compLit(fn, addr.addr, e, typ) // in place
				return
			}
		}
	}
	loc.store(fn, b.expr(fn, e)) // copy value
}

// expr lowers a single-result expression e to SSA form, emitting code
// to fn and returning the Value defined by the expression.
//
func (b *Builder) expr(fn *Function, e ast.Expr) Value {
	if lit := b.constants[e]; lit != nil {
		return lit
	}

	switch e := e.(type) {
	case *ast.BasicLit:
		panic("non-constant BasicLit") // unreachable

	case *ast.FuncLit:
		posn := b.Prog.Files.Position(e.Type.Func)
		fn2 := &Function{
			Name_:     fmt.Sprintf("func@%d.%d", posn.Line, posn.Column),
			Signature: underlyingType(b.exprType(e.Type)).(*types.Signature),
			Pos:       e.Type.Func,
			Enclosing: fn,
			Pkg:       fn.Pkg,
			Prog:      b.Prog,
			syntax: &funcSyntax{
				paramFields:  e.Type.Params,
				resultFields: e.Type.Results,
				body:         e.Body,
			},
		}
		fn.Pkg.AnonFuncs = append(fn.Pkg.AnonFuncs, fn2)
		b.buildFunction(fn2)
		if fn2.FreeVars == nil {
			return fn2
		}
		v := &MakeClosure{Fn: fn2}
		v.setType(b.exprType(e))
		for _, fv := range fn2.FreeVars {
			v.Bindings = append(v.Bindings, fv.Outer)
		}
		return fn.emit(v)

	case *ast.ParenExpr:
		return b.expr(fn, e.X)

	case *ast.TypeAssertExpr: // single-result form only
		v := &TypeAssert{
			X:            b.expr(fn, e.X),
			AssertedType: b.exprType(e),
		}
		v.setType(v.AssertedType)
		return fn.emit(v)

	case *ast.CallExpr:
		typ := b.exprType(e)
		if b.isType(e.Fun) {
			// Type conversion, e.g. string(x) or big.Int(x)
			return emitConv(fn, b.expr(fn, e.Args[0]), typ)
		}
		// Call to "intrinsic" built-ins, e.g. new, make, panic.
		if id, ok := e.Fun.(*ast.Ident); ok {
			obj := b.obj(id)
			if _, ok := fn.Prog.Builtins[obj]; ok {
				if v := b.builtin(fn, id.Name, e.Args, typ); v != nil {
					return v
				}
			}
		}
		// Regular function call.
		var v Call
		b.setCall(fn, e, &v.CallCommon)
		v.setType(typ)
		return fn.emit(&v)

	case *ast.UnaryExpr:
		switch e.Op {
		case token.AND: // &X --- potentially escaping.
			return b.addr(fn, e.X, true).(address).addr
		case token.ADD:
			return b.expr(fn, e.X)
		case token.NOT, token.ARROW, token.SUB, token.XOR: // ! <- - ^
			v := &UnOp{
				Op: e.Op,
				X:  b.expr(fn, e.X),
			}
			v.setType(b.exprType(e))
			return fn.emit(v)
		default:
			panic(e.Op)
		}

	case *ast.BinaryExpr:
		switch e.Op {
		case token.LAND, token.LOR:
			return b.logicalBinop(fn, e)
		case token.SHL, token.SHR:
			fallthrough
		case token.ADD, token.SUB, token.MUL, token.QUO, token.REM, token.AND, token.OR, token.XOR, token.AND_NOT:
			return emitArith(fn, e.Op, b.expr(fn, e.X), b.expr(fn, e.Y), b.exprType(e))

		case token.EQL, token.NEQ, token.GTR, token.LSS, token.LEQ, token.GEQ:
			return emitCompare(fn, e.Op, b.expr(fn, e.X), b.expr(fn, e.Y))
		default:
			panic("illegal op in BinaryExpr: " + e.Op.String())
		}

	case *ast.SliceExpr:
		var low, high Value
		var x Value
		switch underlyingType(b.exprType(e.X)).(type) {
		case *types.Array:
			// Potentially escaping.
			x = b.addr(fn, e.X, true).(address).addr
		case *types.Basic, *types.Slice, *types.Pointer: // *array
			x = b.expr(fn, e.X)
		default:
			unreachable()
		}
		if e.High != nil {
			high = b.expr(fn, e.High)
		}
		if e.Low != nil {
			low = b.expr(fn, e.Low)
		}
		v := &Slice{
			X:    x,
			Low:  low,
			High: high,
		}
		v.setType(b.exprType(e))
		return fn.emit(v)

	case *ast.Ident:
		obj := b.obj(e)
		// Global or universal?
		if v, ok := b.lookup(obj); ok {
			if objKind(obj) == ast.Var {
				v = emitLoad(fn, v) // var (address)
			}
			return v
		}
		// Local?
		return emitLoad(fn, fn.lookup(obj, false)) // var (address)

	case *ast.SelectorExpr:
		// p.M where p is a package.
		if obj := b.isPackageRef(e); obj != nil {
			return b.expr(fn, e.Sel)
		}

		// (*T).f or T.f, the method f from the method-set of type T.
		if b.isType(e.X) {
			id := makeId(e.Sel.Name, fn.Pkg.Types)
			typ := b.exprType(e.X)
			if m := b.Prog.MethodSet(typ)[id]; m != nil {
				return m
			}

			// T must be an interface; return method thunk.
			return makeImethodThunk(b.Prog, typ, id)
		}

		// e.f where e is an expression.
		e, index := b.demoteSelector(e, fn.Pkg)
		switch underlyingType(b.exprType(e.X)).(type) {
		case *types.Struct:
			// Non-addressable struct in a register.
			v := &Field{
				X:     b.expr(fn, e.X),
				Field: index,
			}
			v.setType(b.exprType(e))
			return fn.emit(v)

		case *types.Pointer: // *struct
			// Addressable structs; use FieldAddr and Load.
			return b.addr(fn, e, false).load(fn)
		}

	case *ast.IndexExpr:
		switch t := underlyingType(b.exprType(e.X)).(type) {
		case *types.Array:
			// Non-addressable array (in a register).
			v := &Index{
				X:     b.expr(fn, e.X),
				Index: emitConv(fn, b.expr(fn, e.Index), tInt),
			}
			v.setType(t.Elt)
			return fn.emit(v)

		case *types.Map:
			// Maps are not addressable.
			mapt := underlyingType(b.exprType(e.X)).(*types.Map)
			v := &Lookup{
				X:     b.expr(fn, e.X),
				Index: emitConv(fn, b.expr(fn, e.Index), mapt.Key),
			}
			v.setType(mapt.Elt)
			return fn.emit(v)

		case *types.Basic: // => string
			// Strings are not addressable.
			v := &Lookup{
				X:     b.expr(fn, e.X),
				Index: b.expr(fn, e.Index),
			}
			v.setType(tByte)
			return fn.emit(v)

		case *types.Slice, *types.Pointer: // *array
			// Addressable slice/array; use IndexAddr and Load.
			return b.addr(fn, e, false).load(fn)

		default:
			panic("unexpected container type in IndexExpr: " + t.String())
		}

	case *ast.CompositeLit, *ast.StarExpr:
		// Addressable types (lvalues)
		return b.addr(fn, e, false).load(fn)
	}

	panic(fmt.Sprintf("unexpected expr: %T", e))
}

// stmtList emits to fn code for all statements in list.
func (b *Builder) stmtList(fn *Function, list []ast.Stmt) {
	for _, s := range list {
		b.stmt(fn, s)
	}
}

// setCallFunc populates the function parts of a CallCommon structure
// (Func, Method, Recv, Args[0]) based on the kind of invocation
// occurring in e.
//
func (b *Builder) setCallFunc(fn *Function, e *ast.CallExpr, c *CallCommon) {
	c.Pos = e.Lparen

	// Is the call of the form x.f()?
	sel, ok := noparens(e.Fun).(*ast.SelectorExpr)

	// Case 0: e.Fun evaluates normally to a function.
	if !ok {
		c.Func = b.expr(fn, e.Fun)
		return
	}

	// Case 1: call of form x.F() where x is a package name.
	if obj := b.isPackageRef(sel); obj != nil {
		// This is a specialization of expr(ast.Ident(obj)).
		if v, ok := b.lookup(obj); ok {
			if _, ok := v.(*Function); !ok {
				v = emitLoad(fn, v) // var (address)
			}
			c.Func = v
			return
		}
		panic("undefined package-qualified name: " + obj.GetName())
	}

	// Case 2a: X.f() or (*X).f(): a statically dipatched call to
	// the method f in the method-set of X or *X.  X may be
	// an interface.  Treat like case 0.
	// TODO(adonovan): inline expr() here, to make the call static
	// and to avoid generation of a stub for an interface method.
	if b.isType(sel.X) {
		c.Func = b.expr(fn, e.Fun)
		return
	}

	// Let X be the type of x.
	typ := b.exprType(sel.X)

	// Case 2: x.f(): a statically dispatched call to a method
	// from the method-set of X or perhaps *X (if x is addressable
	// but not a pointer).
	id := makeId(sel.Sel.Name, fn.Pkg.Types)
	// Consult method-set of X.
	if m := b.Prog.MethodSet(typ)[id]; m != nil {
		var recv Value
		aptr := isPointer(typ)
		fptr := isPointer(m.Signature.Recv.Type)
		if aptr == fptr {
			// Actual's and formal's "pointerness" match.
			recv = b.expr(fn, sel.X)
		} else {
			// Actual is a pointer, formal is not.
			// Load a copy.
			recv = emitLoad(fn, b.expr(fn, sel.X))
		}
		c.Func = m
		c.Args = append(c.Args, recv)
		return
	}
	if !isPointer(typ) {
		// Consult method-set of *X.
		if m := b.Prog.MethodSet(pointer(typ))[id]; m != nil {
			// A method found only in MS(*X) must have a
			// pointer formal receiver; but the actual
			// value is not a pointer.
			// Implicit & -- possibly escaping.
			recv := b.addr(fn, sel.X, true).(address).addr
			c.Func = m
			c.Args = append(c.Args, recv)
			return
		}
	}

	switch t := underlyingType(typ).(type) {
	case *types.Struct, *types.Pointer:
		// Case 3: x.f() where x.f is a function value in a
		// struct field f; not a method call.  f is a 'var'
		// (of function type) in the Fields of types.Struct X.
		// Treat like case 0.
		c.Func = b.expr(fn, e.Fun)

	case *types.Interface:
		// Case 4: x.f() where a dynamically dispatched call
		// to an interface method f.  f is a 'func' object in
		// the Methods of types.Interface X
		c.Method, _ = methodIndex(t, t.Methods, id)
		c.Recv = b.expr(fn, sel.X)

	default:
		panic(fmt.Sprintf("illegal (%s).%s() call; X:%T", t, sel.Sel.Name, sel.X))
	}
}

// setCall emits to fn code to evaluate all the parameters of a function
// call e, and populates *c with those values.
//
func (b *Builder) setCall(fn *Function, e *ast.CallExpr, c *CallCommon) {
	// First deal with the f(...) part.
	b.setCallFunc(fn, e, c)

	// Argument passing.  There are 4 cases to consider:
	// 1. Ordinary call to non-variadic function.
	//    All args are treated in the usual manner.
	// 2. Ordinary call to variadic function, f(a, b, v1, ..., vn).
	//    Caller constructs a slice from the varargs arguments v_i.
	// 3. Ellipsis call f(a, b, rest...) to variadic function.
	//    'rest' is already a slice; all args treated in the usual manner.
	// 4. f(g()) where g has >1 return parameters.  f may also be variadic.
	//    TODO(adonovan): implement.

	var args, varargs []ast.Expr = e.Args, nil
	c.HasEllipsis = e.Ellipsis != 0

	// Determine which suffix (if any) of Args is a varargs slice.
	var vt types.Type // element type of the variadic slice
	switch typ := underlyingType(b.exprType(e.Fun)).(type) {
	case *types.Signature:
		np := len(typ.Params)
		if !c.HasEllipsis {
			if typ.IsVariadic && len(args) > np-1 {
				// case 2: ordinary call of variadic function.
				vt = typ.Params[np-1].Type
				args, varargs = args[:np-1], args[np-1:]
			}

			// TODO(adonovan): fix: not disjoint with case above!
			// Consider f(...int) called with g() (int,int)
			// Incomplete.
			if len(args) == 1 && (np > 1 || typ.IsVariadic) {
				if res, ok := b.exprType(args[0]).(*types.Result); ok {
					res = res
					// case 4: f(g()) where g is a multi.
					// TODO(adonovan): generate:
					//   tuple := g()
					//   args = [extract 0, ... extract n]
					//   f(args)
				}
			}
		}

		// Non-varargs.
		for i, arg := range args {
			// TODO(gri): annoyingly Signature.Params
			// doesn't reflect the slice type for a final
			// ...T param.
			t := typ.Params[i].Type
			if typ.IsVariadic && c.HasEllipsis && i == len(args)-1 {
				t = &types.Slice{Elt: t}
			}
			v := emitConv(fn, b.expr(fn, arg), t)
			c.Args = append(c.Args, v)
		}

	default:
		// builtin: ad-hoc typing rules are required for all
		// variadic (append, print, println) and polymorphic
		// (append, copy, delete, close) built-ins.
		//
		// TODO(adonovan): it would so much cleaner if the
		// typechecker would ascribe a unique Signature type
		// to each e.Fun expression calling a built-in.
		// Then we could use the same logic as above.
		//
		// TODO(adonovan): fix: support case 4.  But know that the
		// other tools don't support it for built-ins yet
		// (http://code.google.com/p/go/issues/detail?id=4573),
		// so there's no rush.

		var bptypes []types.Type // formal parameter types of builtins
		switch builtin := e.Fun.(*ast.Ident).Name; builtin {
		case "append":
			// Infer arg types from result type:
			rt := b.exprType(e)
			bptypes = append(bptypes, rt)
			if c.HasEllipsis {
				// 2-arg '...' call form.  No conversions.
				// append([]T, []T) []T
				// append([]byte, string) []byte
			} else {
				// variadic call form.
				// append([]T, ...T) []T
				args, varargs = args[:1], args[1:]
				vt = underlyingType(rt).(*types.Slice).Elt
			}
		case "close":
			// no conv
		case "copy":
			// copy([]T, []T) int
			// Infer arg types from each other.  Sleazy.
			if st, ok := underlyingType(b.exprType(args[0])).(*types.Slice); ok {
				bptypes = append(bptypes, st, st)
			} else if st, ok := underlyingType(b.exprType(args[1])).(*types.Slice); ok {
				bptypes = append(bptypes, st, st)
			} else {
				panic("cannot infer types in call to copy()")
			}
		case "delete":
			// delete(map[K]V, K)
			tkey := underlyingType(b.exprType(args[0])).(*types.Map).Key
			bptypes = append(bptypes, nil)  // map: no conv
			bptypes = append(bptypes, tkey) // key
		case "print", "println": // print{,ln}(any, ...any)
			vt = tEface // variadic
			if !c.HasEllipsis {
				args, varargs = args[:1], args[1:]
			}
		case "len":
			// no conv
		case "cap":
			// no conv
		case "real", "imag":
			// Reverse conversion to "complex" case below.
			// Typechecker, help us out. :(
			var argType types.Type
			switch b.exprType(e).(*types.Basic).Kind {
			case types.UntypedFloat:
				argType = types.Typ[types.UntypedComplex]
			case types.Float64:
				argType = tComplex128
			case types.Float32:
				argType = tComplex64
			default:
				unreachable()
			}
			bptypes = append(bptypes, argType, argType)
		case "complex":
			// Typechecker, help us out. :(
			var argType types.Type
			switch b.exprType(e).(*types.Basic).Kind {
			case types.UntypedComplex:
				argType = types.Typ[types.UntypedFloat]
			case types.Complex128:
				argType = tFloat64
			case types.Complex64:
				argType = tFloat32
			default:
				unreachable()
			}
			bptypes = append(bptypes, argType, argType)
		case "panic":
			bptypes = append(bptypes, tEface)
		case "recover":
			// no-op
		default:
			panic("unknown builtin: " + builtin)
		}

		// Non-varargs.
		for i, arg := range args {
			v := b.expr(fn, arg)
			if i < len(bptypes) && bptypes[i] != nil {
				v = emitConv(fn, v, bptypes[i])
			}
			c.Args = append(c.Args, v)
		}
	}

	// Common code for varargs.
	if len(varargs) > 0 { // case 2
		at := &types.Array{
			Elt: vt,
			Len: int64(len(varargs)),
		}
		a := emitNew(fn, at)
		for i, arg := range varargs {
			iaddr := &IndexAddr{
				X:     a,
				Index: intLiteral(int64(i)),
			}
			iaddr.setType(pointer(vt))
			fn.emit(iaddr)
			emitStore(fn, iaddr, b.expr(fn, arg))
		}
		s := &Slice{X: a}
		s.setType(&types.Slice{Elt: vt})
		c.Args = append(c.Args, fn.emit(s))
	}
}

// assignOp emits to fn code to perform loc += incr or loc -= incr.
func (b *Builder) assignOp(fn *Function, loc lvalue, incr Value, op token.Token) {
	oldv := loc.load(fn)
	loc.store(fn, emitArith(fn, op, oldv, emitConv(fn, incr, oldv.Type()), loc.typ()))
}

// buildGlobal emits code to the g.Pkg.Init function for the variable
// definition(s) of g.  Effects occur out of lexical order; see
// explanation at globalValueSpec.
// Precondition: g == b.globals[obj]
//
func (b *Builder) buildGlobal(g *Global, obj types.Object) {
	spec := g.spec
	if spec == nil {
		return // already built (or in progress)
	}
	b.globalValueSpec(g.Pkg.Init, spec, g, obj)
}

// globalValueSpec emits to init code to define one or all of the vars
// in the package-level ValueSpec spec.
//
// It implements the build phase for a ValueSpec, ensuring that all
// vars are initialized if not already visited by buildGlobal during
// the reference graph traversal.
//
// This function may be called in two modes:
// A) with g and obj non-nil, to initialize just a single global.
//    This occurs during the reference graph traversal.
// B) with g and obj nil, to initialize all globals in the same ValueSpec.
//    This occurs during the left-to-right traversal over the ast.File.
//
// Precondition: g == b.globals[obj]
//
// Package-level var initialization order is quite subtle.
// The side effects of:
//   var a, b = f(), g()
// are not observed left-to-right if b is referenced before a in the
// reference graph traversal.  So, we track which Globals have been
// initialized by setting Global.spec=nil.
//
// Blank identifiers make things more complex since they don't have
// associated types.Objects or ssa.Globals yet we must still ensure
// that their corresponding side effects are observed at the right
// moment.  Consider:
//   var a, _, b = f(), g(), h()
// Here, the relative ordering of the call to g() is unspecified but
// it must occur exactly once, during mode B.  So globalValueSpec for
// blanks must special-case n:n assigments and just evaluate the RHS
// g() for effect.
//
// In a n:1 assignment:
//   var a, _, b = f()
// a reference to either a or b causes both globals to be initialized
// at the same time.  Furthermore, no further work is required to
// ensure that the effects of the blank assignment occur.  We must
// keep track of which n:1 specs have been evaluated, independent of
// which Globals are on the LHS (possibly none, if all are blank).
//
// See also localValueSpec.
//
func (b *Builder) globalValueSpec(init *Function, spec *ast.ValueSpec, g *Global, obj types.Object) {
	switch {
	case len(spec.Values) == len(spec.Names):
		// e.g. var x, y = 0, 1
		// 1:1 assignment.
		// Only the first time for a given GLOBAL has any effect.
		for i, id := range spec.Names {
			var lval lvalue = blank{}
			if g != nil {
				// Mode A: initialized only a single global, g
				if isBlankIdent(id) || b.obj(id) != obj {
					continue
				}
				g.spec = nil
				lval = address{g}
			} else {
				// Mode B: initialize all globals.
				if !isBlankIdent(id) {
					g2 := b.globals[b.obj(id)].(*Global)
					if g2.spec == nil {
						continue // already done
					}
					g2.spec = nil
					lval = address{g2}
				}
			}
			if b.mode&LogSource != 0 {
				fmt.Fprintln(os.Stderr, "build global", id.Name)
			}
			b.exprInPlace(init, lval, spec.Values[i])
			if g != nil {
				break
			}
		}

	case len(spec.Values) == 0:
		// e.g. var x, y int
		// Globals are implicitly zero-initialized.

	default:
		// e.g. var x, _, y = f()
		// n:1 assignment.
		// Only the first time for a given SPEC has any effect.
		if !b.nTo1Vars[spec] {
			b.nTo1Vars[spec] = true
			if b.mode&LogSource != 0 {
				fmt.Fprintln(os.Stderr, "build globals", spec.Names) // ugly...
			}
			tuple := b.exprN(init, spec.Values[0])
			rtypes := tuple.Type().(*types.Result).Values
			for i, id := range spec.Names {
				if !isBlankIdent(id) {
					g := b.globals[b.obj(id)].(*Global)
					g.spec = nil // just an optimization
					emitStore(init, g,
						emitExtract(init, tuple, i, rtypes[i].Type))
				}
			}
		}
	}
}

// localValueSpec emits to fn code to define all of the vars in the
// function-local ValueSpec, spec.
//
// See also globalValueSpec: the two routines are similar but local
// ValueSpecs are much simpler since they are encountered once only,
// in their entirety, in lexical order.
//
func (b *Builder) localValueSpec(fn *Function, spec *ast.ValueSpec) {
	switch {
	case len(spec.Values) == len(spec.Names):
		// e.g. var x, y = 0, 1
		// 1:1 assignment
		for i, id := range spec.Names {
			var lval lvalue = blank{}
			if !isBlankIdent(id) {
				lval = address{fn.addNamedLocal(b.obj(id))}
			}
			b.exprInPlace(fn, lval, spec.Values[i])
		}

	case len(spec.Values) == 0:
		// e.g. var x, y int
		// Locals are implicitly zero-initialized.
		for _, id := range spec.Names {
			if !isBlankIdent(id) {
				fn.addNamedLocal(b.obj(id))
			}
		}

	default:
		// e.g. var x, y = pos()
		tuple := b.exprN(fn, spec.Values[0])
		rtypes := tuple.Type().(*types.Result).Values
		for i, id := range spec.Names {
			if !isBlankIdent(id) {
				lhs := fn.addNamedLocal(b.obj(id))
				emitStore(fn, lhs, emitExtract(fn, tuple, i, rtypes[i].Type))
			}
		}
	}
}

// assignStmt emits code to fn for a parallel assignment of rhss to lhss.
// isDef is true if this is a short variable declaration (:=).
//
// Note the similarity with localValueSpec.
// TODO(adonovan): explain differences.
//
func (b *Builder) assignStmt(fn *Function, lhss, rhss []ast.Expr, isDef bool) {
	// Side effects of all LHSs and RHSs must occur in left-to-right order.
	var lvals []lvalue
	for _, lhs := range lhss {
		var lval lvalue = blank{}
		if !isBlankIdent(lhs) {
			if isDef {
				// Local may be "redeclared" in the same
				// scope, so don't blindly create anew.
				obj := b.obj(lhs.(*ast.Ident))
				if _, ok := fn.objects[obj]; !ok {
					fn.addNamedLocal(obj)
				}
			}
			lval = b.addr(fn, lhs, false) // non-escaping
		}
		lvals = append(lvals, lval)
	}
	if len(lhss) == len(rhss) {
		// e.g. x, y = f(), g()
		if len(lhss) == 1 {
			// x = type{...}
			// Optimization: in-place construction
			// of composite literals.
			b.exprInPlace(fn, lvals[0], rhss[0])
		} else {
			// Parallel assignment.  All reads must occur
			// before all updates, precluding exprInPlace.
			// TODO(adonovan): opt: is it sound to
			// perform exprInPlace if !isDef?
			var rvals []Value
			for _, rval := range rhss {
				rvals = append(rvals, b.expr(fn, rval))
			}
			for i, lval := range lvals {
				lval.store(fn, rvals[i])
			}
		}
	} else {
		// e.g. x, y = pos()
		tuple := b.exprN(fn, rhss[0])
		rtypes := tuple.Type().(*types.Result).Values
		for i, lval := range lvals {
			lval.store(fn, emitExtract(fn, tuple, i, rtypes[i].Type))
		}
	}
}

// compLit emits to fn code to initialize a composite literal e at
// address addr with type typ, typically allocated by Alloc.
// Nested composite literals are recursively initialized in place
// where possible.
//
func (b *Builder) compLit(fn *Function, addr Value, e *ast.CompositeLit, typ types.Type) {
	// TODO(adonovan): test whether typ ever differs from
	// b.exprType(e) and if so document why.

	switch t := underlyingType(typ).(type) {
	case *types.Struct:
		for i, e := range e.Elts {
			// Subtle: field index i is updated in KeyValue case.
			var sf *types.Field
			if kv, ok := e.(*ast.KeyValueExpr); ok {
				fname := kv.Key.(*ast.Ident).Name
				for i, sf = range t.Fields {
					if sf.Name == fname {
						e = kv.Value
						break
					}
				}
			} else {
				sf = t.Fields[i]
			}
			faddr := &FieldAddr{
				X:     addr,
				Field: i,
			}
			faddr.setType(pointer(sf.Type))
			fn.emit(faddr)
			b.exprInPlace(fn, address{faddr}, e)
		}

	case *types.Array, *types.Slice:
		var at *types.Array
		var array Value
		switch t := t.(type) {
		case *types.Slice:
			at = &types.Array{Elt: t.Elt} // set Len later
			array = emitNew(fn, at)
		case *types.Array:
			at = t
			array = addr
		}
		var idx *Literal
		var max int64 = -1
		for _, e := range e.Elts {
			if kv, ok := e.(*ast.KeyValueExpr); ok {
				idx = b.expr(fn, kv.Key).(*Literal)
				e = kv.Value
			} else {
				var idxval int64
				if idx != nil {
					idxval = idx.Int64() + 1
				}
				idx = intLiteral(idxval)
			}
			if idx.Int64() > max {
				max = idx.Int64()
			}
			iaddr := &IndexAddr{
				X:     array,
				Index: idx,
			}
			iaddr.setType(pointer(at.Elt))
			fn.emit(iaddr)
			b.exprInPlace(fn, address{iaddr}, e)
		}
		if t != at { // slice
			at.Len = max + 1
			s := &Slice{X: array}
			s.setType(t)
			emitStore(fn, addr, fn.emit(s))
		}

	case *types.Map:
		m := &MakeMap{Reserve: intLiteral(int64(len(e.Elts)))}
		m.setType(typ)
		emitStore(fn, addr, fn.emit(m))
		for _, e := range e.Elts {
			e := e.(*ast.KeyValueExpr)
			up := &MapUpdate{
				Map:   m,
				Key:   emitConv(fn, b.expr(fn, e.Key), t.Key),
				Value: emitConv(fn, b.expr(fn, e.Value), t.Elt),
			}
			fn.emit(up)
		}

	case *types.Pointer:
		// Pointers can only occur in the recursive case; we
		// strip them off in addr() before calling compLit
		// again, so that we allocate space for a T not a *T.
		panic("compLit(fn, addr, e, *types.Pointer")

	default:
		panic("unexpected CompositeLit type: " + t.String())
	}
}

// switchStmt emits to fn code for the switch statement s, optionally
// labelled by label.
//
func (b *Builder) switchStmt(fn *Function, s *ast.SwitchStmt, label *lblock) {
	// We treat SwitchStmt like a sequential if-else chain.
	// More efficient strategies (e.g. multiway dispatch)
	// are possible if all cases are free of side effects.
	if s.Init != nil {
		b.stmt(fn, s.Init)
	}
	var tag Value = vTrue
	if s.Tag != nil {
		tag = b.expr(fn, s.Tag)
	}
	done := fn.newBasicBlock("switch.done")
	if label != nil {
		label._break = done
	}
	// We pull the default case (if present) down to the end.
	// But each fallthrough label must point to the next
	// body block in source order, so we preallocate a
	// body block (fallthru) for the next case.
	// Unfortunately this makes for a confusing block order.
	var dfltBody *[]ast.Stmt
	var dfltFallthrough *BasicBlock
	var fallthru, dfltBlock *BasicBlock
	ncases := len(s.Body.List)
	for i, clause := range s.Body.List {
		body := fallthru
		if body == nil {
			body = fn.newBasicBlock("switch.body") // first case only
		}

		// Preallocate body block for the next case.
		fallthru = done
		if i+1 < ncases {
			fallthru = fn.newBasicBlock("switch.body")
		}

		cc := clause.(*ast.CaseClause)
		if cc.List == nil {
			// Default case.
			dfltBody = &cc.Body
			dfltFallthrough = fallthru
			dfltBlock = body
			continue
		}

		var nextCond *BasicBlock
		for _, cond := range cc.List {
			nextCond = fn.newBasicBlock("switch.next")
			// TODO(adonovan): opt: when tag==vTrue, we'd
			// get better much code if we use b.cond(cond)
			// instead of BinOp(EQL, tag, b.expr(cond))
			// followed by If.  Don't forget conversions
			// though.
			cond := emitCompare(fn, token.EQL, tag, b.expr(fn, cond))
			emitIf(fn, cond, body, nextCond)
			fn.currentBlock = nextCond
		}
		fn.currentBlock = body
		fn.targets = &targets{
			tail:         fn.targets,
			_break:       done,
			_fallthrough: fallthru,
		}
		b.stmtList(fn, cc.Body)
		fn.targets = fn.targets.tail
		emitJump(fn, done)
		fn.currentBlock = nextCond
	}
	if dfltBlock != nil {
		emitJump(fn, dfltBlock)
		fn.currentBlock = dfltBlock
		fn.targets = &targets{
			tail:         fn.targets,
			_break:       done,
			_fallthrough: dfltFallthrough,
		}
		b.stmtList(fn, *dfltBody)
		fn.targets = fn.targets.tail
	}
	emitJump(fn, done)
	fn.currentBlock = done
}

// typeSwitchStmt emits to fn code for the type switch statement s, optionally
// labelled by label.
//
func (b *Builder) typeSwitchStmt(fn *Function, s *ast.TypeSwitchStmt, label *lblock) {
	// We treat TypeSwitchStmt like a sequential if-else
	// chain.  More efficient strategies (e.g. multiway
	// dispatch) are possible.

	// Typeswitch lowering:
	//
	// var x X
	// switch y := x.(type) {
	// case T1, T2: S1                  // >1 	(y := x)
	// default:     SD                  // 0 types 	(y := x)
	// case T3:     S3                  // 1 type 	(y := x.(T3))
	// }
	//
	//      ...s.Init...
	// 	x := eval x
	//      y := x
	// .caseT1:
	// 	t1, ok1 := typeswitch,ok x <T1>
	// 	if ok1 then goto S1 else goto .caseT2
	// .caseT2:
	// 	t2, ok2 := typeswitch,ok x <T2>
	// 	if ok2 then goto S1 else goto .caseT3
	// .S1:
	// 	...S1...
	// 	goto done
	// .caseT3:
	// 	t3, ok3 := typeswitch,ok x <T3>
	// 	if ok3 then goto S3 else goto default
	// .S3:
	// 	y' := t3  // Kludge: within scope of S3, y resolves here
	// 	...S3...
	// 	goto done
	// .default:
	// 	goto done
	// .done:

	if s.Init != nil {
		b.stmt(fn, s.Init)
	}

	var x, y Value
	var id *ast.Ident
	switch ass := s.Assign.(type) {
	case *ast.ExprStmt: // x.(type)
		x = b.expr(fn, noparens(ass.X).(*ast.TypeAssertExpr).X)
	case *ast.AssignStmt: // y := x.(type)
		x = b.expr(fn, noparens(ass.Rhs[0]).(*ast.TypeAssertExpr).X)
		id = ass.Lhs[0].(*ast.Ident)
		y = fn.addNamedLocal(b.obj(id))
		emitStore(fn, y, x)
	}

	done := fn.newBasicBlock("typeswitch.done")
	if label != nil {
		label._break = done
	}
	var dfltBody []ast.Stmt
	for _, clause := range s.Body.List {
		cc := clause.(*ast.CaseClause)
		if cc.List == nil {
			dfltBody = cc.Body
			continue
		}
		body := fn.newBasicBlock("typeswitch.body")
		var next *BasicBlock
		var casetype types.Type
		var ti Value // t_i, ok := typeassert,ok x <T_i>
		for _, cond := range cc.List {
			next = fn.newBasicBlock("typeswitch.next")
			casetype = b.exprType(cond)
			var condv Value
			if casetype == tUntypedNil {
				condv = emitCompare(fn, token.EQL, x, nilLiteral(x.Type()))
			} else {
				yok := &TypeAssert{
					X:            x,
					AssertedType: casetype,
					CommaOk:      true,
				}
				yok.setType(&types.Result{Values: []*types.Var{
					{Name: "value", Type: casetype},
					varOk,
				}})
				fn.emit(yok)
				ti = emitExtract(fn, yok, 0, casetype)
				condv = emitExtract(fn, yok, 1, tBool)
			}
			emitIf(fn, condv, body, next)
			fn.currentBlock = next
		}
		fn.currentBlock = body
		if id != nil && len(cc.List) == 1 && casetype != tUntypedNil {
			// Declare a new shadow local variable of the
			// same name but a more specific type.
			// Side effect: reassociates binding for y's object.
			y2 := fn.addNamedLocal(b.obj(id))
			y2.Name_ += "'" // debugging aid
			y2.Type_ = pointer(casetype)
			emitStore(fn, y2, ti)
		}
		fn.targets = &targets{
			tail:   fn.targets,
			_break: done,
		}
		b.stmtList(fn, cc.Body)
		fn.targets = fn.targets.tail
		if id != nil {
			fn.objects[b.obj(id)] = y // restore previous y binding
		}
		emitJump(fn, done)
		fn.currentBlock = next
	}
	fn.targets = &targets{
		tail:   fn.targets,
		_break: done,
	}
	b.stmtList(fn, dfltBody)
	fn.targets = fn.targets.tail
	emitJump(fn, done)
	fn.currentBlock = done
}

// selectStmt emits to fn code for the select statement s, optionally
// labelled by label.
//
func (b *Builder) selectStmt(fn *Function, s *ast.SelectStmt, label *lblock) {
	// A blocking select of a single case degenerates to a
	// simple send or receive.
	// TODO(adonovan): is this optimization worth its weight?
	if len(s.Body.List) == 1 {
		clause := s.Body.List[0].(*ast.CommClause)
		if clause.Comm != nil {
			b.stmt(fn, clause.Comm)
			done := fn.newBasicBlock("select.done")
			if label != nil {
				label._break = done
			}
			fn.targets = &targets{
				tail:   fn.targets,
				_break: done,
			}
			b.stmtList(fn, clause.Body)
			fn.targets = fn.targets.tail
			emitJump(fn, done)
			fn.currentBlock = done
			return
		}
	}

	// First evaluate all channels in all cases, and find
	// the directions of each state.
	var states []SelectState
	blocking := true
	for _, clause := range s.Body.List {
		switch comm := clause.(*ast.CommClause).Comm.(type) {
		case nil: // default case
			blocking = false

		case *ast.SendStmt: // ch<- i
			ch := b.expr(fn, comm.Chan)
			states = append(states, SelectState{
				Dir:  ast.SEND,
				Chan: ch,
				Send: emitConv(fn, b.expr(fn, comm.Value),
					underlyingType(ch.Type()).(*types.Chan).Elt),
			})

		case *ast.AssignStmt: // x := <-ch
			states = append(states, SelectState{
				Dir:  ast.RECV,
				Chan: b.expr(fn, noparens(comm.Rhs[0]).(*ast.UnaryExpr).X),
			})

		case *ast.ExprStmt: // <-ch
			states = append(states, SelectState{
				Dir:  ast.RECV,
				Chan: b.expr(fn, noparens(comm.X).(*ast.UnaryExpr).X),
			})
		}
	}

	// We dispatch on the (fair) result of Select using a
	// sequential if-else chain, in effect:
	//
	// idx, recv, recvOk := select(...)
	// if idx == 0 {  // receive on channel 0
	//     x, ok := recv, recvOk
	//     ...state0...
	// } else if v == 1 {   // send on channel 1
	//     ...state1...
	// } else {
	//     ...default...
	// }
	//
	// TODO(adonovan): opt: define and use a multiway dispatch instr.
	pair := &Select{
		States:   states,
		Blocking: blocking,
	}
	pair.setType(tSelect)
	fn.emit(pair)
	idx := emitExtract(fn, pair, 0, tInt)

	done := fn.newBasicBlock("select.done")
	if label != nil {
		label._break = done
	}

	var dfltBody *[]ast.Stmt
	state := 0
	for _, cc := range s.Body.List {
		clause := cc.(*ast.CommClause)
		if clause.Comm == nil {
			dfltBody = &clause.Body
			continue
		}
		body := fn.newBasicBlock("select.body")
		next := fn.newBasicBlock("select.next")
		emitIf(fn, emitCompare(fn, token.EQL, idx, intLiteral(int64(state))), body, next)
		fn.currentBlock = body
		fn.targets = &targets{
			tail:   fn.targets,
			_break: done,
		}
		switch comm := clause.Comm.(type) {
		case *ast.AssignStmt: // x := <-states[state].Chan
			xdecl := fn.addNamedLocal(b.obj(comm.Lhs[0].(*ast.Ident)))

			emitStore(fn, xdecl, emitExtract(fn, pair, 1, indirectType(xdecl.Type())))
			if len(comm.Lhs) == 2 { // x, ok := ...
				okdecl := fn.addNamedLocal(b.obj(comm.Lhs[1].(*ast.Ident)))
				emitStore(fn, okdecl, emitExtract(fn, pair, 2, indirectType(okdecl.Type())))
			}
		}
		b.stmtList(fn, clause.Body)
		fn.targets = fn.targets.tail
		emitJump(fn, done)
		fn.currentBlock = next
		state++
	}
	if dfltBody != nil {
		fn.targets = &targets{
			tail:   fn.targets,
			_break: done,
		}
		b.stmtList(fn, *dfltBody)
		fn.targets = fn.targets.tail
	}
	emitJump(fn, done)
	fn.currentBlock = done
}

// forStmt emits to fn code for the for statement s, optionally
// labelled by label.
//
func (b *Builder) forStmt(fn *Function, s *ast.ForStmt, label *lblock) {
	//	...init...
	//      jump loop
	// loop:
	//      if cond goto body else done
	// body:
	//      ...body...
	//      jump post
	// post:				 (target of continue)
	//      ...post...
	//      jump loop
	// done:                                 (target of break)
	if s.Init != nil {
		b.stmt(fn, s.Init)
	}
	body := fn.newBasicBlock("for.body")
	done := fn.newBasicBlock("for.done") // target of 'break'
	loop := body                         // target of back-edge
	if s.Cond != nil {
		loop = fn.newBasicBlock("for.loop")
	}
	cont := loop // target of 'continue'
	if s.Post != nil {
		cont = fn.newBasicBlock("for.post")
	}
	if label != nil {
		label._break = done
		label._continue = cont
	}
	emitJump(fn, loop)
	fn.currentBlock = loop
	if loop != body {
		b.cond(fn, s.Cond, body, done)
		fn.currentBlock = body
	}
	fn.targets = &targets{
		tail:      fn.targets,
		_break:    done,
		_continue: cont,
	}
	b.stmt(fn, s.Body)
	fn.targets = fn.targets.tail
	emitJump(fn, cont)

	if s.Post != nil {
		fn.currentBlock = cont
		b.stmt(fn, s.Post)
		emitJump(fn, loop) // back-edge
	}
	fn.currentBlock = done
}

// rangeIndexed emits to fn the header for an integer indexed loop
// over array, *array or slice value x.
// The v result is defined only if tv is non-nil.
//
func (b *Builder) rangeIndexed(fn *Function, x Value, tv types.Type) (k, v Value, loop, done *BasicBlock) {
	//
	//      length = len(x)
	//      index = -1
	// loop:                                   (target of continue)
	//      index++
	// 	if index < length goto body else done
	// body:
	//      k = index
	//      v = x[index]
	//      ...body...
	// 	jump loop
	// done:                                   (target of break)

	// Determine number of iterations.
	var length Value
	if arr, ok := deref(x.Type()).(*types.Array); ok {
		// For array or *array, the number of iterations is
		// known statically thanks to the type.  We avoid a
		// data dependence upon x, permitting later dead-code
		// elimination if x is pure, static unrolling, etc.
		// Ranging over a nil *array may have >0 iterations.
		length = intLiteral(arr.Len)
	} else {
		// length = len(x).
		var call Call
		call.Func = b.globals[types.Universe.Lookup("len")]
		call.Args = []Value{x}
		call.setType(tInt)
		length = fn.emit(&call)
	}

	index := fn.addLocal(tInt)
	emitStore(fn, index, intLiteral(-1))

	loop = fn.newBasicBlock("rangeindex.loop")
	emitJump(fn, loop)
	fn.currentBlock = loop

	incr := &BinOp{
		Op: token.ADD,
		X:  emitLoad(fn, index),
		Y:  intLiteral(1),
	}
	incr.setType(tInt)
	emitStore(fn, index, fn.emit(incr))

	body := fn.newBasicBlock("rangeindex.body")
	done = fn.newBasicBlock("rangeindex.done")
	emitIf(fn, emitCompare(fn, token.LSS, incr, length), body, done)
	fn.currentBlock = body

	k = emitLoad(fn, index)
	if tv != nil {
		switch t := underlyingType(x.Type()).(type) {
		case *types.Array:
			instr := &Index{
				X:     x,
				Index: k,
			}
			instr.setType(t.Elt)
			v = fn.emit(instr)

		case *types.Pointer: // *array
			instr := &IndexAddr{
				X:     x,
				Index: k,
			}
			instr.setType(pointer(t.Base.(*types.Array).Elt))
			v = emitLoad(fn, fn.emit(instr))

		case *types.Slice:
			instr := &IndexAddr{
				X:     x,
				Index: k,
			}
			instr.setType(pointer(t.Elt))
			v = emitLoad(fn, fn.emit(instr))

		default:
			panic("rangeIndexed x:" + t.String())
		}
	}
	return
}

// rangeIter emits to fn the header for a loop using
// Range/Next/Extract to iterate over map or string value x.
// tk and tv are the types of the key/value results k and v, or nil
// if the respective component is not wanted.
//
func (b *Builder) rangeIter(fn *Function, x Value, tk, tv types.Type) (k, v Value, loop, done *BasicBlock) {
	//
	//	it = range x
	// loop:                                   (target of continue)
	//	okv = next it                      (ok, key, value)
	//  	ok = extract okv #0
	// 	if ok goto body else done
	// body:
	// 	k = extract okv #1
	// 	v = extract okv #2
	//      ...body...
	// 	jump loop
	// done:                                   (target of break)
	//

	rng := &Range{X: x}
	rng.setType(tRangeIter)
	it := fn.emit(rng)

	loop = fn.newBasicBlock("rangeiter.loop")
	emitJump(fn, loop)
	fn.currentBlock = loop

	okv := &Next{Iter: it}
	okv.setType(&types.Result{Values: []*types.Var{
		varOk,
		{Name: "k", Type: tk},
		{Name: "v", Type: tv},
	}})
	fn.emit(okv)

	body := fn.newBasicBlock("rangeiter.body")
	done = fn.newBasicBlock("rangeiter.done")
	emitIf(fn, emitExtract(fn, okv, 0, tBool), body, done)
	fn.currentBlock = body

	if tk != nil {
		k = emitExtract(fn, okv, 1, tk)
	}
	if tv != nil {
		v = emitExtract(fn, okv, 2, tv)
	}
	return
}

// rangeChan emits to fn the header for a loop that receives from
// channel x until it fails.
// tk is the channel's element type, or nil if the k result is not
// wanted
//
func (b *Builder) rangeChan(fn *Function, x Value, tk types.Type) (k Value, loop, done *BasicBlock) {
	//
	// loop:                                   (target of continue)
	//      ko = <-x                           (key, ok)
	//      ok = extract ko #1
	//      if ok goto body else done
	// body:
	//      k = extract ko #0
	//      ...
	//      goto loop
	// done:                                   (target of break)

	loop = fn.newBasicBlock("rangechan.loop")
	emitJump(fn, loop)
	fn.currentBlock = loop
	recv := &UnOp{
		Op:      token.ARROW,
		X:       x,
		CommaOk: true,
	}
	recv.setType(&types.Result{Values: []*types.Var{
		{Name: "k", Type: tk},
		varOk,
	}})
	ko := fn.emit(recv)
	body := fn.newBasicBlock("rangechan.body")
	done = fn.newBasicBlock("rangechan.done")
	emitIf(fn, emitExtract(fn, ko, 1, tBool), body, done)
	fn.currentBlock = body
	if tk != nil {
		k = emitExtract(fn, ko, 0, tk)
	}
	return
}

// rangeStmt emits to fn code for the range statement s, optionally
// labelled by label.
//
func (b *Builder) rangeStmt(fn *Function, s *ast.RangeStmt, label *lblock) {
	var tk, tv types.Type
	if !isBlankIdent(s.Key) {
		tk = b.exprType(s.Key)
	}
	if s.Value != nil && !isBlankIdent(s.Value) {
		tv = b.exprType(s.Value)
	}

	// If iteration variables are defined (:=), this
	// occurs once outside the loop.
	//
	// Unlike a short variable declaration, a RangeStmt
	// using := never redeclares an existing variable; it
	// always creates a new one.
	if s.Tok == token.DEFINE {
		if tk != nil {
			fn.addNamedLocal(b.obj(s.Key.(*ast.Ident)))
		}
		if tv != nil {
			fn.addNamedLocal(b.obj(s.Value.(*ast.Ident)))
		}
	}

	x := b.expr(fn, s.X)

	var k, v Value
	var loop, done *BasicBlock
	switch rt := underlyingType(x.Type()).(type) {
	case *types.Slice, *types.Array, *types.Pointer: // *array
		k, v, loop, done = b.rangeIndexed(fn, x, tv)

	case *types.Chan:
		k, loop, done = b.rangeChan(fn, x, tk)

	case *types.Map, *types.Basic: // string
		k, v, loop, done = b.rangeIter(fn, x, tk, tv)

	default:
		panic("Cannot range over: " + rt.String())
	}

	// Evaluate both LHS expressions before we update either.
	var kl, vl lvalue
	if tk != nil {
		kl = b.addr(fn, s.Key, false) // non-escaping
	}
	if tv != nil {
		vl = b.addr(fn, s.Value, false) // non-escaping
	}
	if tk != nil {
		kl.store(fn, k)
	}
	if tv != nil {
		vl.store(fn, v)
	}

	if label != nil {
		label._break = done
		label._continue = loop
	}

	fn.targets = &targets{
		tail:      fn.targets,
		_break:    done,
		_continue: loop,
	}
	b.stmt(fn, s.Body)
	fn.targets = fn.targets.tail
	emitJump(fn, loop) // back-edge
	fn.currentBlock = done
}

// stmt lowers statement s to SSA form, emitting code to fn.
func (b *Builder) stmt(fn *Function, _s ast.Stmt) {
	// The label of the current statement.  If non-nil, its _goto
	// target is always set; its _break and _continue are set only
	// within the body of switch/typeswitch/select/for/range.
	// It is effectively an additional default-nil parameter of stmt().
	// TODO(adonovan): fix: handle multiple labels on the same stmt.
	var label *lblock
start:
	switch s := _s.(type) {
	case *ast.EmptyStmt:
		// ignore.  (Usually removed by gofmt.)

	case *ast.DeclStmt: // Con, Var or Typ
		d := s.Decl.(*ast.GenDecl)
		for _, spec := range d.Specs {
			if vs, ok := spec.(*ast.ValueSpec); ok {
				b.localValueSpec(fn, vs)
			}
		}

	case *ast.LabeledStmt:
		label = fn.labelledBlock(s.Label)
		emitJump(fn, label._goto)
		fn.currentBlock = label._goto
		_s = s.Stmt
		goto start // effectively: tailcall stmt(fn, s.Stmt, label)

	case *ast.ExprStmt:
		b.expr(fn, s.X)

	case *ast.SendStmt:
		fn.emit(&Send{
			Chan: b.expr(fn, s.Chan),
			X: emitConv(fn, b.expr(fn, s.Value),
				underlyingType(b.exprType(s.Chan)).(*types.Chan).Elt),
		})

	case *ast.IncDecStmt:
		op := token.ADD
		if s.Tok == token.DEC {
			op = token.SUB
		}
		b.assignOp(fn, b.addr(fn, s.X, false), vOne, op)

	case *ast.AssignStmt:
		switch s.Tok {
		case token.ASSIGN, token.DEFINE:
			b.assignStmt(fn, s.Lhs, s.Rhs, s.Tok == token.DEFINE)

		default: // +=, etc.
			op := s.Tok + token.ADD - token.ADD_ASSIGN
			b.assignOp(fn, b.addr(fn, s.Lhs[0], false), b.expr(fn, s.Rhs[0]), op)
		}

	case *ast.GoStmt:
		// The "intrinsics" new/make/len/cap are forbidden here.
		// panic is treated like an ordinary function call.
		var v Go
		b.setCall(fn, s.Call, &v.CallCommon)
		fn.emit(&v)

	case *ast.DeferStmt:
		// The "intrinsics" new/make/len/cap are forbidden here.
		// panic is treated like an ordinary function call.
		var v Defer
		b.setCall(fn, s.Call, &v.CallCommon)
		fn.emit(&v)

	case *ast.ReturnStmt:
		if fn == fn.Pkg.Init {
			// A "return" within an init block is treated
			// like a "goto" to the next init block.  We
			// use the outermost BREAK target for this purpose.
			var block *BasicBlock
			for t := fn.targets; t != nil; t = t.tail {
				if t._break != nil {
					block = t._break
				}
			}
			emitJump(fn, block)
			fn.currentBlock = fn.newBasicBlock("unreachable")
			return
		}
		var results []Value
		// Per the spec, there are three distinct cases of return.
		switch {
		case len(s.Results) == 0:
			// Return with no arguments.
			// Prior assigns to named result params are
			// reloaded into results tuple.
			// A void function is a degenerate case of this.
			for _, r := range fn.results {
				results = append(results, emitLoad(fn, r))
			}

		case len(s.Results) == 1 && len(fn.Signature.Results) > 1:
			// Return of one expression in a multi-valued function.
			tuple := b.exprN(fn, s.Results[0])
			for i, v := range tuple.Type().(*types.Result).Values {
				results = append(results, emitExtract(fn, tuple, i, v.Type))
			}

		default:
			// Return one or more single-valued expressions.
			// These become the scalar or tuple result.
			for _, r := range s.Results {
				results = append(results, b.expr(fn, r))
			}
		}
		// Perform implicit conversions.
		for i := range results {
			results[i] = emitConv(fn, results[i], fn.Signature.Results[i].Type)
		}
		fn.emit(&Ret{Results: results})
		fn.currentBlock = fn.newBasicBlock("unreachable")

	case *ast.BranchStmt:
		var block *BasicBlock
		switch s.Tok {
		case token.BREAK:
			if s.Label != nil {
				block = fn.labelledBlock(s.Label)._break
			} else {
				for t := fn.targets; t != nil && block == nil; t = t.tail {
					block = t._break
				}
			}

		case token.CONTINUE:
			if s.Label != nil {
				block = fn.labelledBlock(s.Label)._continue
			} else {
				for t := fn.targets; t != nil && block == nil; t = t.tail {
					block = t._continue
				}
			}

		case token.FALLTHROUGH:
			for t := fn.targets; t != nil && block == nil; t = t.tail {
				block = t._fallthrough
			}

		case token.GOTO:
			block = fn.labelledBlock(s.Label)._goto
		}
		if block == nil {
			// TODO(gri): fix: catch these in the typechecker.
			fmt.Printf("ignoring illegal branch: %s %s\n", s.Tok, s.Label)
		} else {
			emitJump(fn, block)
			fn.currentBlock = fn.newBasicBlock("unreachable")
		}

	case *ast.BlockStmt:
		b.stmtList(fn, s.List)

	case *ast.IfStmt:
		if s.Init != nil {
			b.stmt(fn, s.Init)
		}
		then := fn.newBasicBlock("if.then")
		done := fn.newBasicBlock("if.done")
		els := done
		if s.Else != nil {
			els = fn.newBasicBlock("if.else")
		}
		b.cond(fn, s.Cond, then, els)
		fn.currentBlock = then
		b.stmt(fn, s.Body)
		emitJump(fn, done)

		if s.Else != nil {
			fn.currentBlock = els
			b.stmt(fn, s.Else)
			emitJump(fn, done)
		}

		fn.currentBlock = done

	case *ast.SwitchStmt:
		b.switchStmt(fn, s, label)

	case *ast.TypeSwitchStmt:
		b.typeSwitchStmt(fn, s, label)

	case *ast.SelectStmt:
		b.selectStmt(fn, s, label)

	case *ast.ForStmt:
		b.forStmt(fn, s, label)

	case *ast.RangeStmt:
		b.rangeStmt(fn, s, label)

	default:
		panic(fmt.Sprintf("unexpected statement kind: %T", s))
	}
}

// buildFunction builds SSA code for the body of function fn.  Idempotent.
func (b *Builder) buildFunction(fn *Function) {
	if fn.Blocks != nil {
		return // building already started
	}
	if fn.syntax == nil {
		return // not a Go source function
	}
	if fn.syntax.body == nil {
		return // Go source function with no body (external)
	}
	fn.start(b.idents)
	b.stmt(fn, fn.syntax.body)
	if cb := fn.currentBlock; cb != nil && (cb == fn.Blocks[0] || cb.Preds != nil) {
		// We fell off the end: an implicit no-arg return statement.
		fn.emit(new(Ret))
	}
	fn.finish()
}

// memberFromObject populates package pkg with a member for the
// typechecker object obj.
//
// For objects from Go source code, syntax is the associated syntax
// tree (for funcs and vars only); it will be used during the build
// phase.
//
func (b *Builder) memberFromObject(pkg *Package, obj types.Object, syntax ast.Node) {
	name := obj.GetName()
	switch obj := obj.(type) {
	case *types.TypeName:
		pkg.Members[name] = &Type{NamedType: obj.Type.(*types.NamedType)}

	case *types.Const:
		pkg.Members[name] = newLiteral(obj.Val, obj.Type)

	case *types.Var:
		spec, _ := syntax.(*ast.ValueSpec)
		g := &Global{
			Pkg:   pkg,
			Name_: name,
			Type_: pointer(obj.Type), // address
			spec:  spec,
		}
		b.globals[obj] = g
		pkg.Members[name] = g

	case *types.Func:
		var fs *funcSyntax
		var pos token.Pos
		if decl, ok := syntax.(*ast.FuncDecl); ok {
			fs = &funcSyntax{
				recvField:    decl.Recv,
				paramFields:  decl.Type.Params,
				resultFields: decl.Type.Results,
				body:         decl.Body,
			}
			// TODO(gri): make GcImported types.Object
			// implement the full object interface
			// including Pos().  Or at least not crash.
			pos = obj.GetPos()
		}
		sig := obj.Type.(*types.Signature)
		fn := &Function{
			Name_:     name,
			Signature: sig,
			Pos:       pos,
			Pkg:       pkg,
			Prog:      b.Prog,
			syntax:    fs,
		}
		if sig.Recv == nil {
			// Function declaration.
			b.globals[obj] = fn
			pkg.Members[name] = fn
		} else {
			// Method declaration.
			nt := deref(sig.Recv.Type).(*types.NamedType)
			_, method := methodIndex(nt, nt.Methods, makeId(name, pkg.Types))
			b.Prog.concreteMethods[method] = fn
		}

	default: // (incl. *types.Package)
		panic(fmt.Sprintf("unexpected Object type: %T", obj))
	}
}

// membersFromDecl populates package pkg with members for each
// typechecker object (var, func, const or type) associated with the
// specified decl.
//
func (b *Builder) membersFromDecl(pkg *Package, decl ast.Decl) {
	switch decl := decl.(type) {
	case *ast.GenDecl: // import, const, type or var
		switch decl.Tok {
		case token.CONST:
			for _, spec := range decl.Specs {
				for _, id := range spec.(*ast.ValueSpec).Names {
					if !isBlankIdent(id) {
						b.memberFromObject(pkg, b.obj(id), nil)
					}
				}
			}

		case token.VAR:
			for _, spec := range decl.Specs {
				for _, id := range spec.(*ast.ValueSpec).Names {
					if !isBlankIdent(id) {
						b.memberFromObject(pkg, b.obj(id), spec)
					}
				}
			}

		case token.TYPE:
			for _, spec := range decl.Specs {
				id := spec.(*ast.TypeSpec).Name
				if !isBlankIdent(id) {
					b.memberFromObject(pkg, b.obj(id), nil)
				}
			}
		}

	case *ast.FuncDecl:
		id := decl.Name
		if decl.Recv == nil && id.Name == "init" {
			return // init blocks aren't functions
		}
		if !isBlankIdent(id) {
			b.memberFromObject(pkg, b.obj(id), decl)
		}
	}
}

// CreatePackage creates a package from the specified set of files,
// performs type-checking, and allocates all global SSA Values for the
// package.  It returns a new SSA Package providing access to these
// values.
//
// importPath is the full name under which this package is known, such
// as appears in an import declaration. e.g. "sync/atomic".
//
// The ParseFiles() utility may be helpful for parsing a set of Go
// source files.
//
func (b *Builder) CreatePackage(importPath string, files []*ast.File) (*Package, error) {
	typkg, firstErr := b.typechecker.Check(b.Prog.Files, files)
	if firstErr != nil {
		return nil, firstErr
	}
	return b.createPackageImpl(typkg, importPath, files), nil
}

// createPackageImpl constructs an SSA Package from an error-free
// types.Package typkg and populates its Members mapping.  It returns
// the newly constructed ssa.Package.
//
// The real work of building SSA form for each function is not done
// until a subsequent call to BuildPackage.
//
// If files is non-nil, its declarations will be used to generate code
// for functions, methods and init blocks in a subsequent call to
// BuildPackage.  Otherwise, typkg is assumed to have been imported
// from the gc compiler's object files; no code will be available.
//
func (b *Builder) createPackageImpl(typkg *types.Package, importPath string, files []*ast.File) *Package {
	// The typechecker sets types.Package.Path only for GcImported
	// packages, since it doesn't know import path until after typechecking is done.
	// Here we ensure it is always set, since we know the correct path.
	// TODO(adonovan): eliminate redundant ssa.Package.ImportPath field.
	if typkg.Path == "" {
		typkg.Path = importPath
	}

	p := &Package{
		Prog:       b.Prog,
		Types:      typkg,
		ImportPath: importPath,
		Members:    make(map[string]Member),
		files:      files,
	}

	b.packages[typkg] = p
	b.Prog.Packages[importPath] = p

	// CREATE phase.
	// Allocate all package members: vars, funcs and consts and types.
	if len(files) > 0 {
		// Go source package.

		p.Pos = files[0].Package // arbitrary file

		// TODO(gri): make it a typechecker error for there to
		// be duplicate (e.g.) main functions in the same package.
		for _, file := range p.files {
			// ast.Print(b.Prog.Files, file) // debugging
			for _, decl := range file.Decls {
				b.membersFromDecl(p, decl)
			}
		}
	} else {
		// GC-compiled binary package.
		// No code.
		// No position information.

		for _, obj := range p.Types.Scope.Entries {
			b.memberFromObject(p, obj, nil)
		}
	}

	// Compute the method sets
	for _, mem := range p.Members {
		switch t := mem.(type) {
		case *Type:
			t.Methods = b.Prog.MethodSet(t.NamedType)
			t.PtrMethods = b.Prog.MethodSet(pointer(t.NamedType))
		}
	}

	// Add init() function (but not to Members since it can't be referenced).
	p.Init = &Function{
		Name_:     "init",
		Signature: new(types.Signature),
		Pos:       p.Pos,
		Pkg:       p,
		Prog:      b.Prog,
	}

	// Add initializer guard variable.
	initguard := &Global{
		Pkg:   p,
		Name_: "init·guard",
		Type_: pointer(tBool),
	}
	p.Members[initguard.Name()] = initguard

	if b.mode&LogPackages != 0 {
		p.DumpTo(os.Stderr)
	}

	return p
}

// buildDecl builds SSA code for all globals, functions or methods
// declared by decl in package pkg.
//
func (b *Builder) buildDecl(pkg *Package, decl ast.Decl) {
	switch decl := decl.(type) {
	case *ast.GenDecl:
		switch decl.Tok {
		// Nothing to do for CONST, IMPORT.
		case token.VAR:
			for _, spec := range decl.Specs {
				b.globalValueSpec(pkg.Init, spec.(*ast.ValueSpec), nil, nil)
			}
		case token.TYPE:
			for _, spec := range decl.Specs {
				id := spec.(*ast.TypeSpec).Name
				if isBlankIdent(id) {
					continue
				}
				obj := b.obj(id).(*types.TypeName)
				for _, method := range obj.Type.(*types.NamedType).Methods {
					b.buildFunction(b.Prog.concreteMethods[method])
				}
			}
		}

	case *ast.FuncDecl:
		id := decl.Name
		if isBlankIdent(id) {
			// no-op

		} else if decl.Recv == nil && id.Name == "init" {
			// init() block
			if b.mode&LogSource != 0 {
				fmt.Fprintln(os.Stderr, "build init block @", b.Prog.Files.Position(decl.Pos()))
			}
			init := pkg.Init

			// A return statement within an init block is
			// treated like a "goto" to the the next init
			// block, which we stuff in the outermost
			// break label.
			next := init.newBasicBlock("init.next")
			init.targets = &targets{
				tail:   init.targets,
				_break: next,
			}
			b.stmt(init, decl.Body)
			emitJump(init, next)
			init.targets = init.targets.tail
			init.currentBlock = next

		} else if m, ok := b.globals[b.obj(id)]; ok {
			// Package-level function.
			b.buildFunction(m.(*Function))
		}
	}

}

// BuildPackage builds SSA code for all functions and vars in package p.
//
// BuildPackage is idempotent.
//
func (b *Builder) BuildPackage(p *Package) {
	if p.files == nil {
		return // already done (or nothing to do)
	}
	if b.mode&LogSource != 0 {
		fmt.Fprintln(os.Stderr, "build package", p.ImportPath)
	}
	init := p.Init
	init.start(b.idents)

	// Make init() skip if package is already initialized.
	initguard := p.Var("init·guard")
	doinit := init.newBasicBlock("init.start")
	done := init.newBasicBlock("init.done")
	emitIf(init, emitLoad(init, initguard), done, doinit)
	init.currentBlock = doinit
	emitStore(init, initguard, vTrue)

	// TODO(gri): fix: the types.Package.Imports map may contains
	// entries for other package's import statements, if produced
	// by GcImport.  Project it down to just the ones for us.
	imports := make(map[string]*types.Package)
	for _, file := range p.files {
		for _, imp := range file.Imports {
			path, _ := strconv.Unquote(imp.Path.Value)
			if path != "unsafe" {
				imports[path] = p.Types.Imports[path]
			}
		}
	}

	// Call the init() function of each package we import.
	// Order is unspecified (and is in fact nondeterministic).
	for name, imported := range imports {
		p2 := b.packages[imported]
		if p2 == nil {
			panic("Building " + p.Name() + ": CreatePackage has not been called for package " + name)
		}
		// TODO(adonovan): opt: BuildPackage should be
		// package-local, so we can run it for all packages in
		// parallel once CreatePackage has been called for all
		// prerequisites.  Until then, ensure all import
		// dependencies are completely built before we are.
		b.BuildPackage(p2)

		var v Call
		v.Func = p2.Init
		v.Pos = init.Pos
		v.setType(new(types.Result))
		init.emit(&v)
	}

	// Visit the package's var decls and init funcs in source
	// order.  This causes init() code to be generated in
	// topological order.  We visit them transitively through
	// functions of the same package, but we don't treat functions
	// as roots.  TODO(adonovan): fix: don't visit through other
	// packages.
	//
	// We also ensure all functions and methods are built, even if
	// they are unreachable.
	//
	// The order between files is unspecified (and is in fact
	// nondeterministic).
	//
	// TODO(adonovan): the partial order of initialization is
	// underspecified.  Discuss this with gri.
	for _, file := range p.files {
		for _, decl := range file.Decls {
			b.buildDecl(p, decl)
		}
	}
	p.files = nil

	// Finish up.
	emitJump(init, done)
	init.currentBlock = done
	init.emit(new(Ret))
	init.finish()
}
