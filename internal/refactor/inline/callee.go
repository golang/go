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
	"strings"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/typeparams"
)

// A Callee holds information about an inlinable function. Gob-serializable.
type Callee struct {
	impl gobCallee
}

func (callee *Callee) String() string { return callee.impl.Name }

type gobCallee struct {
	Content []byte // file content, compacted to a single func decl

	// results of type analysis (does not reach go/types data structures)
	PkgPath          string       // package path of declaring package
	Name             string       // user-friendly name for error messages
	Unexported       []string     // names of free objects that are unexported
	FreeRefs         []freeRef    // locations of references to free objects
	FreeObjs         []object     // descriptions of free objects
	ValidForCallStmt bool         // function body is "return expr" where expr is f() or <-ch
	NumResults       int          // number of results (according to type, not ast.FieldList)
	Params           []*paramInfo // information about parameters (incl. receiver)
	Results          []*paramInfo // information about result variables
	Effects          []int        // order in which parameters are evaluated (see calleefx)
	HasDefer         bool         // uses defer
	HasBareReturn    bool         // uses bare return in non-void function
	TotalReturns     int          // number of return statements
	TrivialReturns   int          // number of return statements with trivial result conversions
	Labels           []string     // names of all control labels
	Falcon           falconResult // falcon constraint system
}

// A freeRef records a reference to a free object.  Gob-serializable.
// (This means free relative to the FuncDecl as a whole, i.e. excluding parameters.)
type freeRef struct {
	Offset int // byte offset of the reference relative to the FuncDecl
	Object int // index into Callee.freeObjs
}

// An object abstracts a free types.Object referenced by the callee. Gob-serializable.
type object struct {
	Name     string          // Object.Name()
	Kind     string          // one of {var,func,const,type,pkgname,nil,builtin}
	PkgPath  string          // pkgpath of object (or of imported package if kind="pkgname")
	ValidPos bool            // Object.Pos().IsValid()
	Shadow   map[string]bool // names shadowed at one of the object's refs
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

	// TODO(adonovan): support inlining of instantiated generic
	// functions by replacing each occurrence of a type parameter
	// T by its instantiating type argument (e.g. int). We'll need
	// to wrap the instantiating type in parens when it's not an
	// ident or qualified ident to prevent "if x == struct{}"
	// parsing ambiguity, or "T(x)" where T = "*int" or "func()"
	// from misparsing.
	if funcHasTypeParams(decl) {
		return nil, fmt.Errorf("cannot inline generic function %s: type parameters are not yet supported", name)
	}

	// Record the location of all free references in the FuncDecl.
	// (Parameters are not free by this definition.)
	var (
		freeObjIndex = make(map[types.Object]int)
		freeObjs     []object
		freeRefs     []freeRef // free refs that may need renaming
		unexported   []string  // free refs to unexported objects, for later error checks
	)
	var f func(n ast.Node) bool
	visit := func(n ast.Node) { ast.Inspect(n, f) }
	var stack []ast.Node
	stack = append(stack, decl.Type) // for scope of function itself
	f = func(n ast.Node) bool {
		if n != nil {
			stack = append(stack, n) // push
		} else {
			stack = stack[:len(stack)-1] // pop
		}
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
			visit(n.X)
			return false

		case *ast.CompositeLit:
			// Check for struct literals that refer to unexported fields,
			// whether keyed or unkeyed. (Logic assumes well-typedness.)
			litType := deref(info.TypeOf(n))
			if s, ok := typeparams.CoreType(litType).(*types.Struct); ok {
				if n.Type != nil {
					visit(n.Type)
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
					visit(value)
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
						var pkgpath string
						if pkgname, ok := obj.(*types.PkgName); ok {
							pkgpath = pkgname.Imported().Path()
						} else if obj.Pkg() != nil {
							pkgpath = obj.Pkg().Path()
						}
						freeObjs = append(freeObjs, object{
							Name:     obj.Name(),
							Kind:     objectKind(obj),
							PkgPath:  pkgpath,
							ValidPos: obj.Pos().IsValid(),
						})
						freeObjIndex[obj] = objidx
					}

					freeObjs[objidx].Shadow = addShadows(freeObjs[objidx].Shadow, info, obj.Name(), stack)

					freeRefs = append(freeRefs, freeRef{
						Offset: int(n.Pos() - decl.Pos()),
						Object: objidx,
					})
				}
			}
		}
		return true
	}
	visit(decl)

	// Analyze callee body for "return expr" form,
	// where expr is f() or <-ch. These forms are
	// safe to inline as a standalone statement.
	validForCallStmt := false
	if len(decl.Body.List) != 1 {
		// not just a return statement
	} else if ret, ok := decl.Body.List[0].(*ast.ReturnStmt); ok && len(ret.Results) == 1 {
		validForCallStmt = func() bool {
			switch expr := astutil.Unparen(ret.Results[0]).(type) {
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
		hasDefer       = false
		hasBareReturn  = false
		totalReturns   = 0
		trivialReturns = 0
		labels         []string
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
			totalReturns++

			// Are implicit assignment conversions
			// to result variables all trivial?
			trivial := true
			if len(n.Results) > 0 {
				argType := func(i int) types.Type {
					return info.TypeOf(n.Results[i])
				}
				if len(n.Results) == 1 && sig.Results().Len() > 1 {
					// Spread return: return f() where f.Results > 1.
					tuple := info.TypeOf(n.Results[0]).(*types.Tuple)
					argType = func(i int) types.Type {
						return tuple.At(i).Type()
					}
				}
				for i := 0; i < sig.Results().Len(); i++ {
					if !trivialConversion(argType(i), sig.Results().At(i)) {
						trivial = false
						break
					}
				}
			} else if sig.Results().Len() > 0 {
				hasBareReturn = true
			}
			if trivial {
				trivialReturns++
			}
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
		Results:          results,
		Effects:          effects,
		HasDefer:         hasDefer,
		HasBareReturn:    hasBareReturn,
		TotalReturns:     totalReturns,
		TrivialReturns:   trivialReturns,
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
	Name       string          // parameter name (may be blank, or even "")
	Index      int             // index within signature
	IsResult   bool            // false for receiver or parameter, true for result variable
	Assigned   bool            // parameter appears on left side of an assignment statement
	Escapes    bool            // parameter has its address taken
	Refs       []int           // FuncDecl-relative byte offset of parameter ref within body
	Shadow     map[string]bool // names shadowed at one of the above refs
	FalconType string          // name of this parameter's type (if basic) in the falcon system
}

// analyzeParams computes information about parameters of function fn,
// including a simple "address taken" escape analysis.
//
// It returns two new arrays, one of the receiver and parameters, and
// the other of the result variables of function fn.
//
// The input must be well-typed.
func analyzeParams(logf func(string, ...any), fset *token.FileSet, info *types.Info, decl *ast.FuncDecl) (params, results []*paramInfo, effects []int, _ falconResult) {
	fnobj, ok := info.Defs[decl.Name]
	if !ok {
		panic(fmt.Sprintf("%s: no func object for %q",
			fset.PositionFor(decl.Name.Pos(), false), decl.Name)) // ill-typed?
	}

	paramInfos := make(map[*types.Var]*paramInfo)
	{
		sig := fnobj.Type().(*types.Signature)
		newParamInfo := func(param *types.Var, isResult bool) *paramInfo {
			info := &paramInfo{
				Name:     param.Name(),
				IsResult: isResult,
				Index:    len(paramInfos),
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
	var stack []ast.Node
	stack = append(stack, decl.Type) // for scope of function itself
	ast.Inspect(decl.Body, func(n ast.Node) bool {
		if n != nil {
			stack = append(stack, n) // push
		} else {
			stack = stack[:len(stack)-1] // pop
		}

		if id, ok := n.(*ast.Ident); ok {
			if v, ok := info.Uses[id].(*types.Var); ok {
				if pinfo, ok := paramInfos[v]; ok {
					// Record location of ref to parameter/result
					// and any intervening (shadowing) names.
					offset := int(n.Pos() - decl.Pos())
					pinfo.Refs = append(pinfo.Refs, offset)
					pinfo.Shadow = addShadows(pinfo.Shadow, info, pinfo.Name, stack)
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

// -- callee helpers --

// addShadows returns the shadows set augmented by the set of names
// locally shadowed at the location of the reference in the callee
// (identified by the stack). The name of the reference itself is
// excluded.
//
// These shadowed names may not be used in a replacement expression
// for the reference.
func addShadows(shadows map[string]bool, info *types.Info, exclude string, stack []ast.Node) map[string]bool {
	for _, n := range stack {
		if scope := scopeFor(info, n); scope != nil {
			for _, name := range scope.Names() {
				if name != exclude {
					if shadows == nil {
						shadows = make(map[string]bool)
					}
					shadows[name] = true
				}
			}
		}
	}
	return shadows
}

// deref removes a pointer type constructor from the core type of t.
func deref(t types.Type) types.Type {
	if ptr, ok := typeparams.CoreType(t).(*types.Pointer); ok {
		return ptr.Elem()
	}
	return t
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
