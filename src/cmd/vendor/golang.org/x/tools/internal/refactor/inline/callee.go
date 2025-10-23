// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline

// This file defines the analysis of the callee function.

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"slices"
	"strings"

	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/typeparams"
	"golang.org/x/tools/internal/typesinternal"
)

// A Callee holds information about an inlinable function. Gob-serializable.
type Callee struct {
	impl gobCallee
}

func (callee *Callee) String() string { return callee.impl.Name }

type gobCallee struct {
	Content []byte // file content, compacted to a single func decl

	// results of type analysis (does not reach go/types data structures)
	PkgPath          string                 // package path of declaring package
	Name             string                 // user-friendly name for error messages
	Unexported       []string               // names of free objects that are unexported
	FreeRefs         []freeRef              // locations of references to free objects
	FreeObjs         []object               // descriptions of free objects
	ValidForCallStmt bool                   // function body is "return expr" where expr is f() or <-ch
	NumResults       int                    // number of results (according to type, not ast.FieldList)
	Params           []*paramInfo           // information about parameters (incl. receiver)
	TypeParams       []*paramInfo           // information about type parameters
	Results          []*paramInfo           // information about result variables
	Effects          []int                  // order in which parameters are evaluated (see calleefx)
	HasDefer         bool                   // uses defer
	HasBareReturn    bool                   // uses bare return in non-void function
	Returns          [][]returnOperandFlags // metadata about result expressions for each return
	Labels           []string               // names of all control labels
	Falcon           falconResult           // falcon constraint system
}

// returnOperandFlags records metadata about a single result expression in a return
// statement.
type returnOperandFlags int

const (
	nonTrivialResult returnOperandFlags = 1 << iota // return operand has non-trivial conversion to result type
	untypedNilResult                                // return operand is nil literal
)

// A freeRef records a reference to a free object. Gob-serializable.
// (This means free relative to the FuncDecl as a whole, i.e. excluding parameters.)
type freeRef struct {
	Offset int // byte offset of the reference relative to the FuncDecl
	Object int // index into Callee.freeObjs
}

// An object abstracts a free types.Object referenced by the callee. Gob-serializable.
type object struct {
	Name    string // Object.Name()
	Kind    string // one of {var,func,const,type,pkgname,nil,builtin}
	PkgPath string // path of object's package (or imported package if kind="pkgname")
	PkgName string // name of object's package (or imported package if kind="pkgname")
	// TODO(rfindley): should we also track LocalPkgName here? Do we want to
	// preserve the local package name?
	ValidPos bool      // Object.Pos().IsValid()
	Shadow   shadowMap // shadowing info for the object's refs
}

// AnalyzeCallee analyzes a function that is a candidate for inlining
// and returns a Callee that describes it. The Callee object, which is
// serializable, can be passed to one or more subsequent calls to
// Inline, each with a different Caller.
//
// This design allows separate analysis of callers and callees in the
// golang.org/x/tools/go/analysis framework: the inlining information
// about a callee can be recorded as a "fact".
//
// The content should be the actual input to the compiler, not the
// apparent source file according to any //line directives that
// may be present within it.
func AnalyzeCallee(logf func(string, ...any), fset *token.FileSet, pkg *types.Package, info *types.Info, decl *ast.FuncDecl, content []byte) (*Callee, error) {
	checkInfoFields(info)

	// The client is expected to have determined that the callee
	// is a function with a declaration (not a built-in or var).
	fn := info.Defs[decl.Name].(*types.Func)
	sig := fn.Type().(*types.Signature)

	logf("analyzeCallee %v @ %v", fn, fset.PositionFor(decl.Pos(), false))

	// Create user-friendly name ("pkg.Func" or "(pkg.T).Method")
	var name string
	if sig.Recv() == nil {
		name = fmt.Sprintf("%s.%s", fn.Pkg().Name(), fn.Name())
	} else {
		name = fmt.Sprintf("(%s).%s", types.TypeString(sig.Recv().Type(), (*types.Package).Name), fn.Name())
	}

	if decl.Body == nil {
		return nil, fmt.Errorf("cannot inline function %s as it has no body", name)
	}

	// Record the location of all free references in the FuncDecl.
	// (Parameters are not free by this definition.)
	var (
		fieldObjs    = fieldObjs(sig)
		freeObjIndex = make(map[types.Object]int)
		freeObjs     []object
		freeRefs     []freeRef // free refs that may need renaming
		unexported   []string  // free refs to unexported objects, for later error checks
	)
	var f func(n ast.Node, stack []ast.Node) bool
	var stack []ast.Node
	stack = append(stack, decl.Type) // for scope of function itself
	visit := func(n ast.Node, stack []ast.Node) { astutil.PreorderStack(n, stack, f) }
	f = func(n ast.Node, stack []ast.Node) bool {
		switch n := n.(type) {
		case *ast.SelectorExpr:
			// Check selections of free fields/methods.
			if sel, ok := info.Selections[n]; ok &&
				!within(sel.Obj().Pos(), decl) &&
				!n.Sel.IsExported() {
				sym := fmt.Sprintf("(%s).%s", info.TypeOf(n.X), n.Sel.Name)
				unexported = append(unexported, sym)
			}

			// Don't recur into SelectorExpr.Sel.
			visit(n.X, stack)
			return false

		case *ast.CompositeLit:
			// Check for struct literals that refer to unexported fields,
			// whether keyed or unkeyed. (Logic assumes well-typedness.)
			litType := typeparams.Deref(info.TypeOf(n))
			if s, ok := typeparams.CoreType(litType).(*types.Struct); ok {
				if n.Type != nil {
					visit(n.Type, stack)
				}
				for i, elt := range n.Elts {
					var field *types.Var
					var value ast.Expr
					if kv, ok := elt.(*ast.KeyValueExpr); ok {
						field = info.Uses[kv.Key.(*ast.Ident)].(*types.Var)
						value = kv.Value
					} else {
						field = s.Field(i)
						value = elt
					}
					if !within(field.Pos(), decl) && !field.Exported() {
						sym := fmt.Sprintf("(%s).%s", litType, field.Name())
						unexported = append(unexported, sym)
					}

					// Don't recur into KeyValueExpr.Key.
					visit(value, stack)
				}
				return false
			}

		case *ast.Ident:
			if obj, ok := info.Uses[n]; ok {
				// Methods and fields are handled by SelectorExpr and CompositeLit.
				if isField(obj) || isMethod(obj) {
					panic(obj)
				}
				// Inv: id is a lexical reference.

				// A reference to an unexported package-level declaration
				// cannot be inlined into another package.
				if !n.IsExported() &&
					obj.Pkg() != nil && obj.Parent() == obj.Pkg().Scope() {
					unexported = append(unexported, n.Name)
				}

				// Record free reference (incl. self-reference).
				if obj == fn || !within(obj.Pos(), decl) {
					objidx, ok := freeObjIndex[obj]
					if !ok {
						objidx = len(freeObjIndex)
						var pkgPath, pkgName string
						if pn, ok := obj.(*types.PkgName); ok {
							pkgPath = pn.Imported().Path()
							pkgName = pn.Imported().Name()
						} else if obj.Pkg() != nil {
							pkgPath = obj.Pkg().Path()
							pkgName = obj.Pkg().Name()
						}
						freeObjs = append(freeObjs, object{
							Name:     obj.Name(),
							Kind:     objectKind(obj),
							PkgName:  pkgName,
							PkgPath:  pkgPath,
							ValidPos: obj.Pos().IsValid(),
						})
						freeObjIndex[obj] = objidx
					}

					freeObjs[objidx].Shadow = freeObjs[objidx].Shadow.add(info, fieldObjs, obj.Name(), stack)

					freeRefs = append(freeRefs, freeRef{
						Offset: int(n.Pos() - decl.Pos()),
						Object: objidx,
					})
				}
			}
		}
		return true
	}
	visit(decl, stack)

	// Analyze callee body for "return expr" form,
	// where expr is f() or <-ch. These forms are
	// safe to inline as a standalone statement.
	validForCallStmt := false
	if len(decl.Body.List) != 1 {
		// not just a return statement
	} else if ret, ok := decl.Body.List[0].(*ast.ReturnStmt); ok && len(ret.Results) == 1 {
		validForCallStmt = func() bool {
			switch expr := ast.Unparen(ret.Results[0]).(type) {
			case *ast.CallExpr: // f(x)
				callee := typeutil.Callee(info, expr)
				if callee == nil {
					return false // conversion T(x)
				}

				// The only non-void built-in functions that may be
				// called as a statement are copy and recover
				// (though arguably a call to recover should never
				// be inlined as that changes its behavior).
				if builtin, ok := callee.(*types.Builtin); ok {
					return builtin.Name() == "copy" ||
						builtin.Name() == "recover"
				}

				return true // ordinary call f()

			case *ast.UnaryExpr: // <-x
				return expr.Op == token.ARROW // channel receive <-ch
			}

			// No other expressions are valid statements.
			return false
		}()
	}

	// Record information about control flow in the callee
	// (but not any nested functions).
	var (
		hasDefer      = false
		hasBareReturn = false
		returnInfo    [][]returnOperandFlags
		labels        []string
	)
	ast.Inspect(decl.Body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.FuncLit:
			return false // prune traversal
		case *ast.DeferStmt:
			hasDefer = true
		case *ast.LabeledStmt:
			labels = append(labels, n.Label.Name)
		case *ast.ReturnStmt:

			// Are implicit assignment conversions
			// to result variables all trivial?
			var resultInfo []returnOperandFlags
			if len(n.Results) > 0 {
				argInfo := func(i int) (ast.Expr, types.Type) {
					expr := n.Results[i]
					return expr, info.TypeOf(expr)
				}
				if len(n.Results) == 1 && sig.Results().Len() > 1 {
					// Spread return: return f() where f.Results > 1.
					tuple := info.TypeOf(n.Results[0]).(*types.Tuple)
					argInfo = func(i int) (ast.Expr, types.Type) {
						return nil, tuple.At(i).Type()
					}
				}
				for i := range sig.Results().Len() {
					expr, typ := argInfo(i)
					var flags returnOperandFlags
					if typ == types.Typ[types.UntypedNil] { // untyped nil is preserved by go/types
						flags |= untypedNilResult
					}
					if !trivialConversion(info.Types[expr].Value, typ, sig.Results().At(i).Type()) {
						flags |= nonTrivialResult
					}
					resultInfo = append(resultInfo, flags)
				}
			} else if sig.Results().Len() > 0 {
				hasBareReturn = true
			}
			returnInfo = append(returnInfo, resultInfo)
		}
		return true
	})

	// Reject attempts to inline cgo-generated functions.
	for _, obj := range freeObjs {
		// There are others (iconst fconst sconst fpvar macro)
		// but this is probably sufficient.
		if strings.HasPrefix(obj.Name, "_Cfunc_") ||
			strings.HasPrefix(obj.Name, "_Ctype_") ||
			strings.HasPrefix(obj.Name, "_Cvar_") {
			return nil, fmt.Errorf("cannot inline cgo-generated functions")
		}
	}

	// Compact content to just the FuncDecl.
	//
	// As a space optimization, we don't retain the complete
	// callee file content; all we need is "package _; func f() { ... }".
	// This reduces the size of analysis facts.
	//
	// Offsets in the callee information are "relocatable"
	// since they are all relative to the FuncDecl.

	content = append([]byte("package _\n"),
		content[offsetOf(fset, decl.Pos()):offsetOf(fset, decl.End())]...)
	// Sanity check: re-parse the compacted content.
	if _, _, err := parseCompact(content); err != nil {
		return nil, err
	}

	params, results, effects, falcon := analyzeParams(logf, fset, info, decl)
	tparams := analyzeTypeParams(logf, fset, info, decl)
	return &Callee{gobCallee{
		Content:          content,
		PkgPath:          pkg.Path(),
		Name:             name,
		Unexported:       unexported,
		FreeObjs:         freeObjs,
		FreeRefs:         freeRefs,
		ValidForCallStmt: validForCallStmt,
		NumResults:       sig.Results().Len(),
		Params:           params,
		TypeParams:       tparams,
		Results:          results,
		Effects:          effects,
		HasDefer:         hasDefer,
		HasBareReturn:    hasBareReturn,
		Returns:          returnInfo,
		Labels:           labels,
		Falcon:           falcon,
	}}, nil
}

// parseCompact parses a Go source file of the form "package _\n func f() { ... }"
// and returns the sole function declaration.
func parseCompact(content []byte) (*token.FileSet, *ast.FuncDecl, error) {
	fset := token.NewFileSet()
	const mode = parser.ParseComments | parser.SkipObjectResolution | parser.AllErrors
	f, err := parser.ParseFile(fset, "callee.go", content, mode)
	if err != nil {
		return nil, nil, fmt.Errorf("internal error: cannot compact file: %v", err)
	}
	return fset, f.Decls[0].(*ast.FuncDecl), nil
}

// A paramInfo records information about a callee receiver, parameter, or result variable.
type paramInfo struct {
	Name        string    // parameter name (may be blank, or even "")
	Index       int       // index within signature
	IsResult    bool      // false for receiver or parameter, true for result variable
	IsInterface bool      // parameter has a (non-type parameter) interface type
	Assigned    bool      // parameter appears on left side of an assignment statement
	Escapes     bool      // parameter has its address taken
	Refs        []refInfo // information about references to parameter within body
	Shadow      shadowMap // shadowing info for the above refs; see [shadowMap]
	FalconType  string    // name of this parameter's type (if basic) in the falcon system
}

type refInfo struct {
	Offset           int  // FuncDecl-relative byte offset of parameter ref within body
	Assignable       bool // ref appears in context of assignment to known type
	IfaceAssignment  bool // ref is being assigned to an interface
	AffectsInference bool // ref type may affect type inference
	// IsSelectionOperand indicates whether the parameter reference is the
	// operand of a selection (param.f). If so, and param's argument is itself
	// a receiver parameter (a common case), we don't need to desugar (&v or *ptr)
	// the selection: if param.Method is a valid selection, then so is param.fieldOrMethod.
	IsSelectionOperand bool
}

// analyzeParams computes information about parameters of the function declared by decl,
// including a simple "address taken" escape analysis.
//
// It returns two new arrays, one of the receiver and parameters, and
// the other of the result variables of the function.
//
// The input must be well-typed.
func analyzeParams(logf func(string, ...any), fset *token.FileSet, info *types.Info, decl *ast.FuncDecl) (params, results []*paramInfo, effects []int, _ falconResult) {
	sig := signature(fset, info, decl)

	paramInfos := make(map[*types.Var]*paramInfo)
	{
		newParamInfo := func(param *types.Var, isResult bool) *paramInfo {
			info := &paramInfo{
				Name:        param.Name(),
				IsResult:    isResult,
				Index:       len(paramInfos),
				IsInterface: isNonTypeParamInterface(param.Type()),
			}
			paramInfos[param] = info
			return info
		}
		if sig.Recv() != nil {
			params = append(params, newParamInfo(sig.Recv(), false))
		}
		for i := 0; i < sig.Params().Len(); i++ {
			params = append(params, newParamInfo(sig.Params().At(i), false))
		}
		for i := 0; i < sig.Results().Len(); i++ {
			results = append(results, newParamInfo(sig.Results().At(i), true))
		}
	}

	// Search function body for operations &x, x.f(), and x = y
	// where x is a parameter, and record it.
	escape(info, decl, func(v *types.Var, escapes bool) {
		if info := paramInfos[v]; info != nil {
			if escapes {
				info.Escapes = true
			} else {
				info.Assigned = true
			}
		}
	})

	// Record locations of all references to parameters.
	// And record the set of intervening definitions for each parameter.
	//
	// TODO(adonovan): combine this traversal with the one that computes
	// FreeRefs. The tricky part is that calleefx needs this one first.
	fieldObjs := fieldObjs(sig)
	var stack []ast.Node
	stack = append(stack, decl.Type) // for scope of function itself
	astutil.PreorderStack(decl.Body, stack, func(n ast.Node, stack []ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok {
			if v, ok := info.Uses[id].(*types.Var); ok {
				if pinfo, ok := paramInfos[v]; ok {
					// Record ref information, and any intervening (shadowing) names.
					//
					// If the parameter v has an interface type, and the reference id
					// appears in a context where assignability rules apply, there may be
					// an implicit interface-to-interface widening. In that case it is
					// not necessary to insert an explicit conversion from the argument
					// to the parameter's type.
					//
					// Contrapositively, if param is not an interface type, then the
					// assignment may lose type information, for example in the case that
					// the substituted expression is an untyped constant or unnamed type.
					stack = append(stack, n) // (the two calls below want n)
					assignable, ifaceAssign, affectsInference := analyzeAssignment(info, stack)
					ref := refInfo{
						Offset:             int(n.Pos() - decl.Pos()),
						Assignable:         assignable,
						IfaceAssignment:    ifaceAssign,
						AffectsInference:   affectsInference,
						IsSelectionOperand: isSelectionOperand(stack),
					}
					pinfo.Refs = append(pinfo.Refs, ref)
					pinfo.Shadow = pinfo.Shadow.add(info, fieldObjs, pinfo.Name, stack)
				}
			}
		}
		return true
	})

	// Compute subset and order of parameters that are strictly evaluated.
	// (Depends on Refs computed above.)
	effects = calleefx(info, decl.Body, paramInfos)
	logf("effects list = %v", effects)

	falcon := falcon(logf, fset, paramInfos, info, decl)

	return params, results, effects, falcon
}

// analyzeTypeParams computes information about the type parameters of the function declared by decl.
func analyzeTypeParams(_ logger, fset *token.FileSet, info *types.Info, decl *ast.FuncDecl) []*paramInfo {
	sig := signature(fset, info, decl)
	paramInfos := make(map[*types.TypeName]*paramInfo)
	var params []*paramInfo
	collect := func(tpl *types.TypeParamList) {
		for i := range tpl.Len() {
			typeName := tpl.At(i).Obj()
			info := &paramInfo{Name: typeName.Name()}
			params = append(params, info)
			paramInfos[typeName] = info
		}
	}
	collect(sig.RecvTypeParams())
	collect(sig.TypeParams())

	// Find references.
	// We don't care about most of the properties that matter for parameter references:
	// a type is immutable, cannot have its address taken, and does not undergo conversions.
	// TODO(jba): can we nevertheless combine this with the traversal in analyzeParams?
	var stack []ast.Node
	stack = append(stack, decl.Type) // for scope of function itself
	astutil.PreorderStack(decl.Body, stack, func(n ast.Node, stack []ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok {
			if v, ok := info.Uses[id].(*types.TypeName); ok {
				if pinfo, ok := paramInfos[v]; ok {
					ref := refInfo{Offset: int(n.Pos() - decl.Pos())}
					pinfo.Refs = append(pinfo.Refs, ref)
					pinfo.Shadow = pinfo.Shadow.add(info, nil, pinfo.Name, stack)
				}
			}
		}
		return true
	})
	return params
}

func signature(fset *token.FileSet, info *types.Info, decl *ast.FuncDecl) *types.Signature {
	fnobj, ok := info.Defs[decl.Name]
	if !ok {
		panic(fmt.Sprintf("%s: no func object for %q",
			fset.PositionFor(decl.Name.Pos(), false), decl.Name)) // ill-typed?
	}
	return fnobj.Type().(*types.Signature)
}

// -- callee helpers --

// analyzeAssignment looks at the given stack, and analyzes certain
// attributes of the innermost expression.
//
// In all cases we 'fail closed' when we cannot detect (or for simplicity
// choose not to detect) the condition in question, meaning we err on the side
// of the more restrictive rule. This is noted for each result below.
//
//   - assignable reports whether the expression is used in a position where
//     assignability rules apply, such as in an actual assignment, as call
//     argument, or in a send to a channel. Defaults to 'false'. If assignable
//     is false, the other two results are irrelevant.
//   - ifaceAssign reports whether that assignment is to an interface type.
//     This is important as we want to preserve the concrete type in that
//     assignment. Defaults to 'true'. Notably, if the assigned type is a type
//     parameter, we assume that it could have interface type.
//   - affectsInference is (somewhat vaguely) defined as whether or not the
//     type of the operand may affect the type of the surrounding syntax,
//     through type inference. It is infeasible to completely reverse engineer
//     type inference, so we over approximate: if the expression is an argument
//     to a call to a generic function (but not method!) that uses type
//     parameters, assume that unification of that argument may affect the
//     inferred types.
func analyzeAssignment(info *types.Info, stack []ast.Node) (assignable, ifaceAssign, affectsInference bool) {
	remaining, parent, expr := exprContext(stack)
	if parent == nil {
		return false, false, false
	}

	// TODO(golang/go#70638): simplify when types.Info records implicit conversions.

	// Types do not need to match for assignment to a variable.
	if assign, ok := parent.(*ast.AssignStmt); ok {
		for i, v := range assign.Rhs {
			if v == expr {
				if i >= len(assign.Lhs) {
					return false, false, false // ill typed
				}
				// Check to see if the assignment is to an interface type.
				if i < len(assign.Lhs) {
					// TODO: We could handle spread calls here, but in current usage expr
					// is an ident.
					if id, _ := assign.Lhs[i].(*ast.Ident); id != nil && info.Defs[id] != nil {
						// Types must match for a defining identifier in a short variable
						// declaration.
						return false, false, false
					}
					// In all other cases, types should be known.
					typ := info.TypeOf(assign.Lhs[i])
					return true, typ == nil || types.IsInterface(typ), false
				}
				// Default:
				return assign.Tok == token.ASSIGN, true, false
			}
		}
	}

	// Types do not need to match for an initializer with known type.
	if spec, ok := parent.(*ast.ValueSpec); ok && spec.Type != nil {
		if slices.Contains(spec.Values, expr) {
			typ := info.TypeOf(spec.Type)
			return true, typ == nil || types.IsInterface(typ), false
		}
	}

	// Types do not need to match for index expressions.
	if ix, ok := parent.(*ast.IndexExpr); ok {
		if ix.Index == expr {
			typ := info.TypeOf(ix.X)
			if typ == nil {
				return true, true, false
			}
			m, _ := typeparams.CoreType(typ).(*types.Map)
			return true, m == nil || types.IsInterface(m.Key()), false
		}
	}

	// Types do not need to match for composite literal keys, values, or
	// fields.
	if kv, ok := parent.(*ast.KeyValueExpr); ok {
		var under types.Type
		if len(remaining) > 0 {
			if complit, ok := remaining[len(remaining)-1].(*ast.CompositeLit); ok {
				if typ := info.TypeOf(complit); typ != nil {
					// Unpointer to allow for pointers to slices or arrays, which are
					// permitted as the types of nested composite literals without a type
					// name.
					under = typesinternal.Unpointer(typeparams.CoreType(typ))
				}
			}
		}
		if kv.Key == expr { // M{expr: ...}: assign to map key
			m, _ := under.(*types.Map)
			return true, m == nil || types.IsInterface(m.Key()), false
		}
		if kv.Value == expr {
			switch under := under.(type) {
			case interface{ Elem() types.Type }: // T{...: expr}: assign to map/array/slice element
				return true, types.IsInterface(under.Elem()), false
			case *types.Struct: // Struct{k: expr}
				if id, _ := kv.Key.(*ast.Ident); id != nil {
					for fi := range under.NumFields() {
						field := under.Field(fi)
						if info.Uses[id] == field {
							return true, types.IsInterface(field.Type()), false
						}
					}
				}
			default:
				return true, true, false
			}
		}
	}
	if lit, ok := parent.(*ast.CompositeLit); ok {
		for i, v := range lit.Elts {
			if v == expr {
				typ := info.TypeOf(lit)
				if typ == nil {
					return true, true, false
				}
				// As in the KeyValueExpr case above, unpointer to handle pointers to
				// array/slice literals.
				under := typesinternal.Unpointer(typeparams.CoreType(typ))
				switch under := under.(type) {
				case interface{ Elem() types.Type }: // T{expr}: assign to map/array/slice element
					return true, types.IsInterface(under.Elem()), false
				case *types.Struct: // Struct{expr}: assign to unkeyed struct field
					if i < under.NumFields() {
						return true, types.IsInterface(under.Field(i).Type()), false
					}
				}
				return true, true, false
			}
		}
	}

	// Types do not need to match for values sent to a channel.
	if send, ok := parent.(*ast.SendStmt); ok {
		if send.Value == expr {
			typ := info.TypeOf(send.Chan)
			if typ == nil {
				return true, true, false
			}
			ch, _ := typeparams.CoreType(typ).(*types.Chan)
			return true, ch == nil || types.IsInterface(ch.Elem()), false
		}
	}

	// Types do not need to match for an argument to a call, unless the
	// corresponding parameter has type parameters, as in that case the
	// argument type may affect inference.
	if call, ok := parent.(*ast.CallExpr); ok {
		if _, ok := isConversion(info, call); ok {
			return false, false, false // redundant conversions are handled at the call site
		}
		// Ordinary call. Could be a call of a func, builtin, or function value.
		for i, arg := range call.Args {
			if arg == expr {
				typ := info.TypeOf(call.Fun)
				if typ == nil {
					return true, true, false
				}
				sig, _ := typeparams.CoreType(typ).(*types.Signature)
				if sig != nil {
					// Find the relevant parameter type, accounting for variadics.
					paramType := paramTypeAtIndex(sig, call, i)
					ifaceAssign := paramType == nil || types.IsInterface(paramType)
					affectsInference := false
					if fn := typeutil.StaticCallee(info, call); fn != nil {
						if sig2 := fn.Type().(*types.Signature); sig2.Recv() == nil {
							originParamType := paramTypeAtIndex(sig2, call, i)
							affectsInference = originParamType == nil || new(typeparams.Free).Has(originParamType)
						}
					}
					return true, ifaceAssign, affectsInference
				}
			}
		}
	}

	return false, false, false
}

// paramTypeAtIndex returns the effective parameter type at the given argument
// index in call, if valid.
func paramTypeAtIndex(sig *types.Signature, call *ast.CallExpr, index int) types.Type {
	if plen := sig.Params().Len(); sig.Variadic() && index >= plen-1 && !call.Ellipsis.IsValid() {
		if s, ok := sig.Params().At(plen - 1).Type().(*types.Slice); ok {
			return s.Elem()
		}
	} else if index < plen {
		return sig.Params().At(index).Type()
	}
	return nil // ill typed
}

// exprContext returns the innermost parent->child expression nodes for the
// given outer-to-inner stack, after stripping parentheses, along with the
// remaining stack up to the parent node.
//
// If no such context exists, returns (nil, nil, nil).
func exprContext(stack []ast.Node) (remaining []ast.Node, parent ast.Node, expr ast.Expr) {
	expr, _ = stack[len(stack)-1].(ast.Expr)
	if expr == nil {
		return nil, nil, nil
	}
	i := len(stack) - 2
	for ; i >= 0; i-- {
		if pexpr, ok := stack[i].(*ast.ParenExpr); ok {
			expr = pexpr
		} else {
			parent = stack[i]
			break
		}
	}
	if parent == nil {
		return nil, nil, nil
	}
	// inv: i is the index of parent in the stack.
	return stack[:i], parent, expr
}

// isSelectionOperand reports whether the innermost node of stack is operand
// (x) of a selection x.f.
func isSelectionOperand(stack []ast.Node) bool {
	_, parent, expr := exprContext(stack)
	if parent == nil {
		return false
	}
	sel, ok := parent.(*ast.SelectorExpr)
	return ok && sel.X == expr
}

// A shadowMap records information about shadowing at any of the parameter's
// references within the callee decl.
//
// For each name shadowed at a reference to the parameter within the callee
// body, shadow map records the 1-based index of the callee decl parameter
// causing the shadowing, or -1, if the shadowing is not due to a callee decl.
// A value of zero (or missing) indicates no shadowing. By convention,
// self-shadowing is excluded from the map.
//
// For example, in the following callee
//
//	func f(a, b int) int {
//		c := 2 + b
//		return a + c
//	}
//
// the shadow map of a is {b: 2, c: -1}, because b is shadowed by the 2nd
// parameter. The shadow map of b is {a: 1}, because c is not shadowed at the
// use of b.
type shadowMap map[string]int

// add returns the [shadowMap] augmented by the set of names
// locally shadowed at the location of the reference in the callee
// (identified by the stack). The name of the reference itself is
// excluded.
//
// These shadowed names may not be used in a replacement expression
// for the reference.
func (s shadowMap) add(info *types.Info, paramIndexes map[types.Object]int, exclude string, stack []ast.Node) shadowMap {
	for _, n := range stack {
		if scope := scopeFor(info, n); scope != nil {
			for _, name := range scope.Names() {
				if name != exclude {
					if s == nil {
						s = make(shadowMap)
					}
					obj := scope.Lookup(name)
					if idx, ok := paramIndexes[obj]; ok {
						s[name] = idx + 1
					} else {
						s[name] = -1
					}
				}
			}
		}
	}
	return s
}

// fieldObjs returns a map of each types.Object defined by the given signature
// to its index in the parameter list. Parameters with missing or blank name
// are skipped.
func fieldObjs(sig *types.Signature) map[types.Object]int {
	m := make(map[types.Object]int)
	for i := range sig.Params().Len() {
		if p := sig.Params().At(i); p.Name() != "" && p.Name() != "_" {
			m[p] = i
		}
	}
	return m
}

func isField(obj types.Object) bool {
	if v, ok := obj.(*types.Var); ok && v.IsField() {
		return true
	}
	return false
}

func isMethod(obj types.Object) bool {
	if f, ok := obj.(*types.Func); ok && f.Type().(*types.Signature).Recv() != nil {
		return true
	}
	return false
}

// -- serialization --

var (
	_ gob.GobEncoder = (*Callee)(nil)
	_ gob.GobDecoder = (*Callee)(nil)
)

func (callee *Callee) GobEncode() ([]byte, error) {
	var out bytes.Buffer
	if err := gob.NewEncoder(&out).Encode(callee.impl); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}

func (callee *Callee) GobDecode(data []byte) error {
	return gob.NewDecoder(bytes.NewReader(data)).Decode(&callee.impl)
}
