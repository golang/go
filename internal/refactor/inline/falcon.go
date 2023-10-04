// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline

// This file defines the callee side of the "fallible constant" analysis.

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/format"
	"go/token"
	"go/types"
	"strconv"
	"strings"

	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/typeparams"
)

// falconResult is the result of the analysis of the callee.
type falconResult struct {
	Types       []falconType // types for falcon constraint environment
	Constraints []string     // constraints (Go expressions) on values of fallible constants
}

// A falconType specifies the name and underlying type of a synthetic
// defined type for use in falcon constraints.
//
// Unique types from callee code are bijectively mapped onto falcon
// types so that constraints are independent of callee type
// information but preserve type equivalence classes.
//
// Fresh names are deliberately obscure to avoid shadowing even if a
// callee parameter has a nanme like "int" or "any".
type falconType struct {
	Name string
	Kind types.BasicKind // string/number/bool
}

// falcon identifies "fallible constant" expressions, which are
// expressions that may fail to compile if one or more of their
// operands is changed from non-constant to constant.
//
// Consider:
//
//	func sub(s string, i, j int) string { return s[i:j] }
//
// If parameters are replaced by constants, the compiler is
// required to perform these additional checks:
//
//   - if i is constant, 0 <= i.
//   - if s and i are constant, i <= len(s).
//   - ditto for j.
//   - if i and j are constant, i <= j.
//
// s[i:j] is thus a "fallible constant" expression dependent on {s, i,
// j}. Each falcon creates a set of conditional constraints across one
// or more parameter variables.
//
//   - When inlining a call such as sub("abc", -1, 2), the parameter i
//     cannot be eliminated by substitution as its argument value is
//     negative.
//
//   - When inlining sub("", 2, 1), all three parameters cannot be be
//     simultaneously eliminated by substitution without violating i
//     <= len(s) and j <= len(s), but the parameters i and j could be
//     safely eliminated without s.
//
// Parameters that cannot be eliminated must remain non-constant,
// either in the form of a binding declaration:
//
//	{ var i int = -1; return "abc"[i:2] }
//
// or a parameter of a literalization:
//
//	func (i int) string { return "abc"[i:2] }(-1)
//
// These example expressions are obviously doomed to fail at run
// time, but in realistic cases such expressions are dominated by
// appropriate conditions that make them reachable only when safe:
//
//	if 0 <= i && i <= j && j <= len(s) { _ = s[i:j] }
//
// (In principle a more sophisticated inliner could entirely eliminate
// such unreachable blocks based on the condition being always-false
// for the given parameter substitution, but this is tricky to do safely
// because the type-checker considers only a single configuration.
// Consider: if runtime.GOOS == "linux" { ... }.)
//
// We believe this is an exhaustive list of "fallible constant" operations:
//
//   - switch z { case x: case y } 	// duplicate case values
//   - s[i], s[i:j], s[i:j:k]		// index out of bounds (0 <= i <= j <= k <= len(s))
//   - T{x: 0}				// index out of bounds, duplicate index
//   - x/y, x%y, x/=y, x%=y		// integer division by zero; minint/-1 overflow
//   - x+y, x-y, x*y			// arithmetic overflow
//   - x<<y				// shift out of range
//   - -x				// negation of minint
//   - T(x)				// value out of range
//
// The fundamental reason for this elaborate algorithm is that the
// "separate analysis" of callee and caller, as required when running
// in an environment such as unitchecker, means that there is no way
// for us to simply invoke the type checker on the combination of
// caller and callee code, as by the time we analyze the caller, we no
// longer have access to type information for the callee (and, in
// particular, any of its direct dependencies that are not direct
// dependencies of the caller). So, in effect, we are forced to map
// the problem in a neutral (callee-type-independent) constraint
// system that can be verified later.
func falcon(logf func(string, ...any), fset *token.FileSet, params map[*types.Var]*paramInfo, info *types.Info, decl *ast.FuncDecl) falconResult {

	st := &falconState{
		logf:   logf,
		fset:   fset,
		params: params,
		info:   info,
		decl:   decl,
	}

	// type mapping
	st.int = st.typename(types.Typ[types.Int])
	st.any = "interface{}" // don't use "any" as it may be shadowed
	for obj, info := range st.params {
		if isBasic(obj.Type(), types.IsConstType) {
			info.FalconType = st.typename(obj.Type())
		}
	}

	st.stmt(st.decl.Body)

	return st.result
}

type falconState struct {
	// inputs
	logf   func(string, ...any)
	fset   *token.FileSet
	params map[*types.Var]*paramInfo
	info   *types.Info
	decl   *ast.FuncDecl

	// working state
	int       string
	any       string
	typenames typeutil.Map

	result falconResult
}

// typename returns the name in the falcon constraint system
// of a given string/number/bool type t. Falcon types are
// specified directly in go/types data structures rather than
// by name, avoiding potential shadowing conflicts with
// confusing parameter names such as "int".
//
// Also, each distinct type (as determined by types.Identical)
// is mapped to a fresh type in the falcon system so that we
// can map the types in the callee code into a neutral form
// that does not depend on imports, allowing us to detect
// potential conflicts such as
//
//	map[any]{T1(1): 0, T2(1): 0}
//
// where T1=T2.
func (st *falconState) typename(t types.Type) string {
	name, ok := st.typenames.At(t).(string)
	if !ok {
		basic := t.Underlying().(*types.Basic)

		// That dot ۰ is an Arabic zero numeral U+06F0.
		// It is very unlikely to appear in a real program.
		// TODO(adonovan): use a non-heuristic solution.
		name = fmt.Sprintf("%s۰%d", basic, st.typenames.Len())
		st.typenames.Set(t, name)
		st.logf("falcon: emit type %s %s // %q", name, basic, t)
		st.result.Types = append(st.result.Types, falconType{
			Name: name,
			Kind: basic.Kind(),
		})
	}
	return name
}

// -- constraint emission --

// emit emits a Go expression that must have a legal type.
// In effect, we let the go/types constant folding algorithm
// do most of the heavy lifting (though it may be hard to
// believe from the complexity of this algorithm!).
func (st *falconState) emit(constraint ast.Expr) {
	var out strings.Builder
	if err := format.Node(&out, st.fset, constraint); err != nil {
		panic(err) // can't happen
	}
	syntax := out.String()
	st.logf("falcon: emit constraint %s", syntax)
	st.result.Constraints = append(st.result.Constraints, syntax)
}

// emitNonNegative emits an []T{}[index] constraint,
// which ensures index is non-negative if constant.
func (st *falconState) emitNonNegative(index ast.Expr) {
	st.emit(&ast.IndexExpr{
		X: &ast.CompositeLit{
			Type: &ast.ArrayType{
				Elt: makeIdent(st.int),
			},
		},
		Index: index,
	})
}

// emitMonotonic emits an []T{}[i:j] constraint,
// which ensures i <= j if both are constant.
func (st *falconState) emitMonotonic(i, j ast.Expr) {
	st.emit(&ast.SliceExpr{
		X: &ast.CompositeLit{
			Type: &ast.ArrayType{
				Elt: makeIdent(st.int),
			},
		},
		Low:  i,
		High: j,
	})
}

// emitUnique emits a T{elem1: 0, ... elemN: 0} constraint,
// which ensures that all constant elems are unique.
// T may be a map, slice, or array depending
// on the desired check semantics.
func (st *falconState) emitUnique(typ ast.Expr, elems []ast.Expr) {
	if len(elems) > 1 {
		var elts []ast.Expr
		for _, elem := range elems {
			elts = append(elts, &ast.KeyValueExpr{
				Key:   elem,
				Value: makeIntLit(0),
			})
		}
		st.emit(&ast.CompositeLit{
			Type: typ,
			Elts: elts,
		})
	}
}

// -- traversal --

// The traversal functions scan the callee body for expressions that
// are not constant but would become constant if the parameter vars
// were redeclared as constants, and emits for each one a constraint
// (a Go expression) with the property that it will not type-check
// (using types.CheckExpr) if the particular argument values are
// unsuitable.
//
// These constraints are checked by Inline with the actual
// constant argument values. Violations cause it to reject
// parameters as candidates for substitution.

func (st *falconState) stmt(s ast.Stmt) {
	ast.Inspect(s, func(n ast.Node) bool {
		switch n := n.(type) {
		case ast.Expr:
			_ = st.expr(n)
			return false // skip usual traversal

		case *ast.AssignStmt:
			switch n.Tok {
			case token.QUO_ASSIGN, token.REM_ASSIGN:
				// x /= y
				// Possible "integer division by zero"
				// Emit constraint: 1/y.
				_ = st.expr(n.Lhs[0])
				kY := st.expr(n.Rhs[0])
				if kY, ok := kY.(ast.Expr); ok {
					op := token.QUO
					if n.Tok == token.REM_ASSIGN {
						op = token.REM
					}
					st.emit(&ast.BinaryExpr{
						Op: op,
						X:  makeIntLit(1),
						Y:  kY,
					})
				}
				return false // skip usual traversal
			}

		case *ast.SwitchStmt:
			if n.Init != nil {
				st.stmt(n.Init)
			}
			tBool := types.Type(types.Typ[types.Bool])
			tagType := tBool // default: true
			if n.Tag != nil {
				st.expr(n.Tag)
				tagType = st.info.TypeOf(n.Tag)
			}

			// Possible "duplicate case value".
			// Emit constraint map[T]int{v1: 0, ..., vN:0}
			// to ensure all maybe-constant case values are unique
			// (unless switch tag is boolean, which is relaxed).
			var unique []ast.Expr
			for _, clause := range n.Body.List {
				clause := clause.(*ast.CaseClause)
				for _, caseval := range clause.List {
					if k := st.expr(caseval); k != nil {
						unique = append(unique, st.toExpr(k))
					}
				}
				for _, stmt := range clause.Body {
					st.stmt(stmt)
				}
			}
			if unique != nil && !types.Identical(tagType.Underlying(), tBool) {
				tname := st.any
				if !types.IsInterface(tagType) {
					tname = st.typename(tagType)
				}
				t := &ast.MapType{
					Key:   makeIdent(tname),
					Value: makeIdent(st.int),
				}
				st.emitUnique(t, unique)
			}
		}
		return true
	})
}

// fieldTypes visits the .Type of each field in the list.
func (st *falconState) fieldTypes(fields *ast.FieldList) {
	if fields != nil {
		for _, field := range fields.List {
			_ = st.expr(field.Type)
		}
	}
}

// expr visits the expression (or type) and returns a
// non-nil result if the expression is constant or would
// become constant if all suitable function parameters were
// redeclared as constants.
//
// If the expression is constant, st.expr returns its type
// and value (types.TypeAndValue). If the expression would
// become constant, st.expr returns an ast.Expr tree whose
// leaves are literals and parameter references, and whose
// interior nodes are operations that may become constant,
// such as -x, x+y, f(x), and T(x). We call these would-be
// constant expressions "fallible constants", since they may
// fail to type-check for some values of x, i, and j. (We
// refer to the non-nil cases collectively as "maybe
// constant", and the nil case as "definitely non-constant".)
//
// As a side effect, st.expr emits constraints for each
// fallible constant expression; this is its main purpose.
//
// Consequently, st.expr must visit the entire subtree so
// that all necessary constraints are emitted. It may not
// short-circuit the traversal when it encounters a constant
// subexpression as constants may contain arbitrary other
// syntax that may impose constraints. Consider (as always)
// this contrived but legal example of a type parameter (!)
// that contains statement syntax:
//
//	func f[T [unsafe.Sizeof(func() { stmts })]int]()
//
// There is no need to emit constraints for (e.g.) s[i] when s
// and i are already constants, because we know the expression
// is sound, but it is sometimes easier to emit these
// redundant constraints than to avoid them.
func (st *falconState) expr(e ast.Expr) (res any) { // = types.TypeAndValue | ast.Expr
	tv := st.info.Types[e]
	if tv.Value != nil {
		// A constant value overrides any other result.
		defer func() { res = tv }()
	}

	switch e := e.(type) {
	case *ast.Ident:
		if v, ok := st.info.Uses[e].(*types.Var); ok {
			if _, ok := st.params[v]; ok && isBasic(v.Type(), types.IsConstType) {
				return e // reference to constable parameter
			}
		}
		// (References to *types.Const are handled by the defer.)

	case *ast.BasicLit:
		// constant

	case *ast.ParenExpr:
		return st.expr(e.X)

	case *ast.FuncLit:
		_ = st.expr(e.Type)
		st.stmt(e.Body)
		// definitely non-constant

	case *ast.CompositeLit:
		// T{k: v, ...}, where T ∈ {array,*array,slice,map},
		// imposes a constraint that all constant k are
		// distinct and, for arrays [n]T, within range 0-n.
		//
		// Types matter, not just values. For example,
		// an interface-keyed map may contain keys
		// that are numerically equal so long as they
		// are of distinct types. For example:
		//
		//   type myint int
		//   map[any]bool{1: true, 1:        true} // error: duplicate key
		//   map[any]bool{1: true, int16(1): true} // ok
		//   map[any]bool{1: true, myint(1): true} // ok
		//
		// This can be asserted by emitting a
		// constraint of the form T{k1: 0, ..., kN: 0}.
		if e.Type != nil {
			_ = st.expr(e.Type)
		}
		t := deref(typeparams.CoreType(deref(tv.Type)))
		var uniques []ast.Expr
		for _, elt := range e.Elts {
			if kv, ok := elt.(*ast.KeyValueExpr); ok {
				if !is[*types.Struct](t) {
					if k := st.expr(kv.Key); k != nil {
						uniques = append(uniques, st.toExpr(k))
					}
				}
				_ = st.expr(kv.Value)
			} else {
				_ = st.expr(elt)
			}
		}
		if uniques != nil {
			// Inv: not a struct.

			// The type T in constraint T{...} depends on the CompLit:
			// - for a basic-keyed map, use map[K]int;
			// - for an interface-keyed map, use map[any]int;
			// - for a slice, use []int;
			// - for an array or *array, use [n]int.
			// The last two entail progressively stronger index checks.
			var ct ast.Expr // type syntax for constraint
			switch t := t.(type) {
			case *types.Map:
				if types.IsInterface(t.Key()) {
					ct = &ast.MapType{
						Key:   makeIdent(st.any),
						Value: makeIdent(st.int),
					}
				} else {
					ct = &ast.MapType{
						Key:   makeIdent(st.typename(t.Key())),
						Value: makeIdent(st.int),
					}
				}
			case *types.Array: // or *array
				ct = &ast.ArrayType{
					Len: makeIntLit(t.Len()),
					Elt: makeIdent(st.int),
				}
			default:
				panic(t)
			}
			st.emitUnique(ct, uniques)
		}
		// definitely non-constant

	case *ast.SelectorExpr:
		_ = st.expr(e.X)
		_ = st.expr(e.Sel)
		// The defer is sufficient to handle
		// qualified identifiers (pkg.Const).
		// All other cases are definitely non-constant.

	case *ast.IndexExpr:
		if tv.IsType() {
			// type C[T]
			_ = st.expr(e.X)
			_ = st.expr(e.Index)
		} else {
			// term x[i]
			//
			// Constraints (if x is slice/string/array/*array, not map):
			// - i >= 0
			//     if i is a fallible constant
			// - i < len(x)
			//     if x is array/*array and
			//     i is a fallible constant;
			//  or if s is a string and both i,
			//     s are maybe-constants,
			//     but not both are constants.
			kX := st.expr(e.X)
			kI := st.expr(e.Index)
			if kI != nil && !is[*types.Map](st.info.TypeOf(e.X).Underlying()) {
				if kI, ok := kI.(ast.Expr); ok {
					st.emitNonNegative(kI)
				}
				// Emit constraint to check indices against known length.
				// TODO(adonovan): factor with SliceExpr logic.
				var x ast.Expr
				if kX != nil {
					// string
					x = st.toExpr(kX)
				} else if arr, ok := deref(st.info.TypeOf(e.X).Underlying()).(*types.Array); ok {
					// array, *array
					x = &ast.CompositeLit{
						Type: &ast.ArrayType{
							Len: makeIntLit(arr.Len()),
							Elt: makeIdent(st.int),
						},
					}
				}
				if x != nil {
					st.emit(&ast.IndexExpr{
						X:     x,
						Index: st.toExpr(kI),
					})
				}
			}
		}
		// definitely non-constant

	case *ast.SliceExpr:
		// x[low:high:max]
		//
		// Emit non-negative constraints for each index,
		// plus low <= high <= max <= len(x)
		// for each pair that are maybe-constant
		// but not definitely constant.

		kX := st.expr(e.X)
		var kLow, kHigh, kMax any
		if e.Low != nil {
			kLow = st.expr(e.Low)
			if kLow != nil {
				if kLow, ok := kLow.(ast.Expr); ok {
					st.emitNonNegative(kLow)
				}
			}
		}
		if e.High != nil {
			kHigh = st.expr(e.High)
			if kHigh != nil {
				if kHigh, ok := kHigh.(ast.Expr); ok {
					st.emitNonNegative(kHigh)
				}
				if kLow != nil {
					st.emitMonotonic(st.toExpr(kLow), st.toExpr(kHigh))
				}
			}
		}
		if e.Max != nil {
			kMax = st.expr(e.Max)
			if kMax != nil {
				if kMax, ok := kMax.(ast.Expr); ok {
					st.emitNonNegative(kMax)
				}
				if kHigh != nil {
					st.emitMonotonic(st.toExpr(kHigh), st.toExpr(kMax))
				}
			}
		}

		// Emit constraint to check indices against known length.
		var x ast.Expr
		if kX != nil {
			// string
			x = st.toExpr(kX)
		} else if arr, ok := deref(st.info.TypeOf(e.X).Underlying()).(*types.Array); ok {
			// array, *array
			x = &ast.CompositeLit{
				Type: &ast.ArrayType{
					Len: makeIntLit(arr.Len()),
					Elt: makeIdent(st.int),
				},
			}
		}
		if x != nil {
			// Avoid slice[::max] if kHigh is nonconstant (nil).
			high, max := st.toExpr(kHigh), st.toExpr(kMax)
			if high == nil {
				high = max // => slice[:max:max]
			}
			st.emit(&ast.SliceExpr{
				X:    x,
				Low:  st.toExpr(kLow),
				High: high,
				Max:  max,
			})
		}
		// definitely non-constant

	case *ast.TypeAssertExpr:
		_ = st.expr(e.X)
		if e.Type != nil {
			_ = st.expr(e.Type)
		}

	case *ast.CallExpr:
		_ = st.expr(e.Fun)
		if tv, ok := st.info.Types[e.Fun]; ok && tv.IsType() {
			// conversion T(x)
			//
			// Possible "value out of range".
			kX := st.expr(e.Args[0])
			if kX != nil && isBasic(tv.Type, types.IsConstType) {
				conv := convert(makeIdent(st.typename(tv.Type)), st.toExpr(kX))
				if is[ast.Expr](kX) {
					st.emit(conv)
				}
				return conv
			}
			return nil // definitely non-constant
		}

		// call f(x)

		all := true // all args are possibly-constant
		kArgs := make([]ast.Expr, len(e.Args))
		for i, arg := range e.Args {
			if kArg := st.expr(arg); kArg != nil {
				kArgs[i] = st.toExpr(kArg)
			} else {
				all = false
			}
		}

		// Calls to built-ins with fallibly constant arguments
		// may become constant. All other calls are either
		// constant or non-constant
		if id, ok := e.Fun.(*ast.Ident); ok && all && tv.Value == nil {
			if builtin, ok := st.info.Uses[id].(*types.Builtin); ok {
				switch builtin.Name() {
				case "len", "imag", "real", "complex", "min", "max":
					return &ast.CallExpr{
						Fun:      id,
						Args:     kArgs,
						Ellipsis: e.Ellipsis,
					}
				}
			}
		}

	case *ast.StarExpr: // *T, *ptr
		_ = st.expr(e.X)

	case *ast.UnaryExpr:
		// + - ! ^ & <- ~
		//
		// Possible "negation of minint".
		// Emit constraint: -x
		kX := st.expr(e.X)
		if kX != nil && !is[types.TypeAndValue](kX) {
			if e.Op == token.SUB {
				st.emit(&ast.UnaryExpr{
					Op: e.Op,
					X:  st.toExpr(kX),
				})
			}

			return &ast.UnaryExpr{
				Op: e.Op,
				X:  st.toExpr(kX),
			}
		}

	case *ast.BinaryExpr:
		kX := st.expr(e.X)
		kY := st.expr(e.Y)
		switch e.Op {
		case token.QUO, token.REM:
			// x/y, x%y
			//
			// Possible "integer division by zero" or
			// "minint / -1" overflow.
			// Emit constraint: x/y or 1/y
			if kY != nil {
				if kX == nil {
					kX = makeIntLit(1)
				}
				st.emit(&ast.BinaryExpr{
					Op: e.Op,
					X:  st.toExpr(kX),
					Y:  st.toExpr(kY),
				})
			}

		case token.ADD, token.SUB, token.MUL:
			// x+y, x-y, x*y
			//
			// Possible "arithmetic overflow".
			// Emit constraint: x+y
			if kX != nil && kY != nil {
				st.emit(&ast.BinaryExpr{
					Op: e.Op,
					X:  st.toExpr(kX),
					Y:  st.toExpr(kY),
				})
			}

		case token.SHL, token.SHR:
			// x << y, x >> y
			//
			// Possible "constant shift too large".
			// Either operand may be too large individually,
			// and they may be too large together.
			// Emit constraint:
			//    x << y (if both maybe-constant)
			//    x << 0 (if y is non-constant)
			//    1 << y (if x is non-constant)
			if kX != nil || kY != nil {
				x := st.toExpr(kX)
				if x == nil {
					x = makeIntLit(1)
				}
				y := st.toExpr(kY)
				if y == nil {
					y = makeIntLit(0)
				}
				st.emit(&ast.BinaryExpr{
					Op: e.Op,
					X:  x,
					Y:  y,
				})
			}

		case token.LSS, token.GTR, token.EQL, token.NEQ, token.LEQ, token.GEQ:
			// < > == != <= <=
			//
			// A "x cmp y" expression with constant operands x, y is
			// itself constant, but I can't see how a constant bool
			// could be fallible: the compiler doesn't reject duplicate
			// boolean cases in a switch, presumably because boolean
			// switches are less like n-way branches and more like
			// sequential if-else chains with possibly overlapping
			// conditions; and there is (sadly) no way to convert a
			// boolean constant to an int constant.
		}
		if kX != nil && kY != nil {
			return &ast.BinaryExpr{
				Op: e.Op,
				X:  st.toExpr(kX),
				Y:  st.toExpr(kY),
			}
		}

	// types
	//
	// We need to visit types (and even type parameters)
	// in order to reach all the places where things could go wrong:
	//
	// 	const (
	// 		s = ""
	// 		i = 0
	// 	)
	// 	type C[T [unsafe.Sizeof(func() { _ = s[i] })]int] bool

	case *ast.IndexListExpr:
		_ = st.expr(e.X)
		for _, expr := range e.Indices {
			_ = st.expr(expr)
		}

	case *ast.Ellipsis:
		if e.Elt != nil {
			_ = st.expr(e.Elt)
		}

	case *ast.ArrayType:
		if e.Len != nil {
			_ = st.expr(e.Len)
		}
		_ = st.expr(e.Elt)

	case *ast.StructType:
		st.fieldTypes(e.Fields)

	case *ast.FuncType:
		st.fieldTypes(e.TypeParams)
		st.fieldTypes(e.Params)
		st.fieldTypes(e.Results)

	case *ast.InterfaceType:
		st.fieldTypes(e.Methods)

	case *ast.MapType:
		_ = st.expr(e.Key)
		_ = st.expr(e.Value)

	case *ast.ChanType:
		_ = st.expr(e.Value)
	}
	return
}

// toExpr converts the result of visitExpr to a falcon expression.
// (We don't do this in visitExpr as we first need to discriminate
// constants from maybe-constants.)
func (st *falconState) toExpr(x any) ast.Expr {
	switch x := x.(type) {
	case nil:
		return nil

	case types.TypeAndValue:
		lit := makeLiteral(x.Value)
		if !isBasic(x.Type, types.IsUntyped) {
			// convert to "typed" type
			lit = &ast.CallExpr{
				Fun:  makeIdent(st.typename(x.Type)),
				Args: []ast.Expr{lit},
			}
		}
		return lit

	case ast.Expr:
		return x

	default:
		panic(x)
	}
}

func makeLiteral(v constant.Value) ast.Expr {
	switch v.Kind() {
	case constant.Bool:
		// Rather than refer to the true or false built-ins,
		// which could be shadowed by poorly chosen parameter
		// names, we use 0 == 0 for true and 0 != 0 for false.
		op := token.EQL
		if !constant.BoolVal(v) {
			op = token.NEQ
		}
		return &ast.BinaryExpr{
			Op: op,
			X:  makeIntLit(0),
			Y:  makeIntLit(0),
		}

	case constant.String:
		return &ast.BasicLit{
			Kind:  token.STRING,
			Value: v.ExactString(),
		}

	case constant.Int:
		return &ast.BasicLit{
			Kind:  token.INT,
			Value: v.ExactString(),
		}

	case constant.Float:
		return &ast.BasicLit{
			Kind:  token.FLOAT,
			Value: v.ExactString(),
		}

	case constant.Complex:
		// The components could be float or int.
		y := makeLiteral(constant.Imag(v))
		y.(*ast.BasicLit).Value += "i" // ugh
		if re := constant.Real(v); !consteq(re, kZeroInt) {
			// complex: x + yi
			y = &ast.BinaryExpr{
				Op: token.ADD,
				X:  makeLiteral(re),
				Y:  y,
			}
		}
		return y

	default:
		panic(v.Kind())
	}
}

func makeIntLit(x int64) *ast.BasicLit {
	return &ast.BasicLit{
		Kind:  token.INT,
		Value: strconv.FormatInt(x, 10),
	}
}

func isBasic(t types.Type, info types.BasicInfo) bool {
	basic, ok := t.Underlying().(*types.Basic)
	return ok && basic.Info()&info != 0
}
