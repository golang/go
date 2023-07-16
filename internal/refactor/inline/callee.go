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

	// syntax derived from compacted Content (not serialized)
	fset *token.FileSet
	decl *ast.FuncDecl

	// results of type analysis (does not reach go/types data structures)
	PkgPath          string    // package path of declaring package
	Name             string    // user-friendly name for error messages
	Unexported       []string  // names of free objects that are unexported
	FreeRefs         []freeRef // locations of references to free objects
	FreeObjs         []object  // descriptions of free objects
	BodyIsReturnExpr bool      // function body is "return expr(s)"
	ValidForCallStmt bool      // => bodyIsReturnExpr and sole expr is f() or <-ch
	NumResults       int       // number of results (according to type, not ast.FieldList)
}

// A freeRef records a reference to a free object.  Gob-serializable.
type freeRef struct {
	Start, End int // Callee.content[start:end] is extent of the reference
	Object     int // index into Callee.freeObjs
}

// An object abstracts a free types.Object referenced by the callee. Gob-serializable.
type object struct {
	Name     string // Object.Name()
	Kind     string // one of {var,func,const,type,pkgname,nil,builtin}
	PkgPath  string // pkgpath of object (or of imported package if kind="pkgname")
	ValidPos bool   // Object.Pos().IsValid()
}

func (callee *gobCallee) offset(pos token.Pos) int { return offsetOf(callee.fset, pos) }

// AnalyzeCallee analyzes a function that is a candidate for inlining
// and returns a Callee that describes it. The Callee object, which is
// serializable, can be passed to one or more subsequent calls to
// Inline, each with a different Caller.
//
// This design allows separate analysis of callers and callees in the
// golang.org/x/tools/go/analysis framework: the inlining information
// about a callee can be recorded as a "fact".
func AnalyzeCallee(fset *token.FileSet, pkg *types.Package, info *types.Info, decl *ast.FuncDecl, content []byte) (*Callee, error) {

	// The client is expected to have determined that the callee
	// is a function with a declaration (not a built-in or var).
	fn := info.Defs[decl.Name].(*types.Func)
	sig := fn.Type().(*types.Signature)

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
	if decl.Type.TypeParams != nil {
		return nil, fmt.Errorf("cannot inline generic function %s: type parameters are not yet supported", name)
	}

	// Record the location of all free references in the callee body.
	var (
		freeObjIndex = make(map[types.Object]int)
		freeObjs     []object
		freeRefs     []freeRef // free refs that may need renaming
		unexported   []string  // free refs to unexported objects, for later error checks
	)
	var visit func(n ast.Node) bool
	visit = func(n ast.Node) bool {
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

				// Record free reference.
				if !within(obj.Pos(), decl) {
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
					freeRefs = append(freeRefs, freeRef{
						Start:  offsetOf(fset, n.Pos()),
						End:    offsetOf(fset, n.End()),
						Object: objidx,
					})
				}
			}
		}
		return true
	}
	ast.Inspect(decl, visit)

	// Analyze callee body for "return results" form, where
	// results is one or more expressions or an n-ary call.
	validForCallStmt := false
	bodyIsReturnExpr := decl.Type.Results != nil && len(decl.Type.Results.List) > 0 &&
		len(decl.Body.List) == 1 &&
		is[*ast.ReturnStmt](decl.Body.List[0]) &&
		len(decl.Body.List[0].(*ast.ReturnStmt).Results) > 0
	if bodyIsReturnExpr {
		ret := decl.Body.List[0].(*ast.ReturnStmt)

		// Ascertain whether the results expression(s)
		// would be safe to inline as a standalone statement.
		// (This is true only for a single call or receive expression.)
		validForCallStmt = func() bool {
			if len(ret.Results) == 1 {
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
			}

			// No other expressions are valid statements.
			return false
		}()
	}

	// As a space optimization, we don't retain the complete
	// callee file content; all we need is "package _; func f() { ... }".
	// This reduces the size of analysis facts.
	//
	// The FileSet file/line info is no longer meaningful
	// and should not be used in error messages.
	// But the FileSet offsets are valid w.r.t. the content.
	//
	// (For ease of debugging we could insert a //line directive after
	// the package decl but it seems more trouble than it's worth.)
	{
		start, end := offsetOf(fset, decl.Pos()), offsetOf(fset, decl.End())

		var compact bytes.Buffer
		compact.WriteString("package _\n")
		compact.Write(content[start:end])
		content = compact.Bytes()

		// Re-parse the compacted content.
		var err error
		decl, err = parseCompact(fset, content)
		if err != nil {
			return nil, err
		}

		// (content, decl) are now updated.

		// Adjust the freeRefs offsets.
		delta := int(offsetOf(fset, decl.Pos()) - start)
		for i := range freeRefs {
			freeRefs[i].Start += delta
			freeRefs[i].End += delta
		}
	}

	return &Callee{gobCallee{
		Content:          content,
		fset:             fset,
		decl:             decl,
		PkgPath:          pkg.Path(),
		Name:             name,
		Unexported:       unexported,
		FreeObjs:         freeObjs,
		FreeRefs:         freeRefs,
		BodyIsReturnExpr: bodyIsReturnExpr,
		ValidForCallStmt: validForCallStmt,
		NumResults:       sig.Results().Len(),
	}}, nil
}

// parseCompact parses a Go source file of the form "package _\n func f() { ... }"
// and returns the sole function declaration.
func parseCompact(fset *token.FileSet, content []byte) (*ast.FuncDecl, error) {
	const mode = parser.ParseComments | parser.SkipObjectResolution | parser.AllErrors
	f, err := parser.ParseFile(fset, "callee.go", content, mode)
	if err != nil {
		return nil, fmt.Errorf("internal error: cannot compact file: %v", err)
	}
	return f.Decls[0].(*ast.FuncDecl), nil
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
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&callee.impl); err != nil {
		return err
	}
	fset := token.NewFileSet()
	decl, err := parseCompact(fset, callee.impl.Content)
	if err != nil {
		return err
	}
	callee.impl.fset = fset
	callee.impl.decl = decl
	return nil
}
