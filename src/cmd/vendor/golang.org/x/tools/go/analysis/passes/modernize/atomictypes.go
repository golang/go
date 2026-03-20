// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modernize

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/edge"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	typeindexanalyzer "golang.org/x/tools/internal/analysis/typeindex"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/goplsexport"
	"golang.org/x/tools/internal/refactor"
	"golang.org/x/tools/internal/typesinternal"
	"golang.org/x/tools/internal/typesinternal/typeindex"
	"golang.org/x/tools/internal/versions"
)

var atomicTypesAnalyzer = &analysis.Analyzer{
	Name: "atomictypes",
	Doc:  analyzerutil.MustExtractDoc(doc, "atomictypes"),
	Requires: []*analysis.Analyzer{
		inspect.Analyzer,
		typeindexanalyzer.Analyzer,
	},
	Run: runAtomic,
	URL: "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/modernize#atomictypes",
}

func init() {
	// Export to gopls until this is a published modernizer.
	goplsexport.AtomicTypesModernizer = atomicTypesAnalyzer
}

// TODO(mkalil): support the Pointer variants.
// Consider the following function signatures for pointer loading:
// func LoadPointer(addr *unsafe.Pointer) (val unsafe.Pointer)
// func (x *Pointer[T]) Load() *T
// Since the former uses *unsafe.Pointer while the latter uses *Pointer[T],
// we would need to determine the type T to apply the transformation, and there
// will be additional edits required to remove any *unsafe.Pointer casts.
// 	"LoadPointer", "StorePointer", "SwapPointer", "CompareAndSwapPointer"

// sync/atomic functions of interest. Some added in go1.19, some added in go1.23.
var syncAtomicFuncs = []string{
	// Added in go1.19.
	"AddInt32", "AddInt64", "AddUint32", "AddUint64", "AddUintptr",
	"CompareAndSwapInt32", "CompareAndSwapInt64", "CompareAndSwapUint32", "CompareAndSwapUint64", "CompareAndSwapUintptr",
	"LoadInt32", "LoadInt64", "LoadUint32", "LoadUint64", "LoadUintptr",
	"StoreInt32", "StoreInt64", "StoreUint32", "StoreUint64", "StoreUintptr",
	"SwapInt32", "SwapInt64", "SwapUint32", "SwapUint64", "SwapUintptr",
	// Added in go1.23.
	"AndInt32", "AndInt64", "AndUint32", "AndUint64", "AndUintptr",
	"OrInt32", "OrInt64", "OrUint32", "OrUint64", "OrUintptr",
}

func runAtomic(pass *analysis.Pass) (any, error) {
	if !typesinternal.Imports(pass.Pkg, "sync/atomic") {
		return nil, nil // doesn't directly import sync/atomic
	}

	var (
		index = pass.ResultOf[typeindexanalyzer.Analyzer].(*typeindex.Index)
		info  = pass.TypesInfo
	)

	// Gather all candidate variables v appearing
	// in calls to atomic.AddInt32(&v, ...) et al.
	var (
		atomicPkg *types.Package
		vars      = make(map[*types.Var]string) // maps candidate vars v to the name of the call they appear in
	)
	for _, funcName := range syncAtomicFuncs {
		obj := index.Object("sync/atomic", funcName)
		if obj == nil {
			continue
		}
		atomicPkg = obj.Pkg()
		for curCall := range index.Calls(obj) {
			call := curCall.Node().(*ast.CallExpr)
			if unary, ok := call.Args[0].(*ast.UnaryExpr); ok && unary.Op == token.AND {
				var v *types.Var
				switch x := unary.X.(type) {
				case *ast.Ident:
					v, _ = info.Uses[x].(*types.Var)
				case *ast.SelectorExpr:
					if seln, ok := info.Selections[x]; ok {
						v, _ = seln.Obj().(*types.Var)
					}
				}
				if v != nil && !v.Exported() {
					// v must be a non-exported package or local var, or a struct field.
					switch v.Kind() {
					case types.RecvVar, types.ParamVar, types.ResultVar:
						continue // fix would change func signature
					}
					vars[v] = funcName
				}
			}
		}
	}

	// Check that all uses of each candidate variable
	// appear in calls of the form atomic.AddInt32(&v, ...).
nextvar:
	for v, funcName := range vars {
		var edits []analysis.TextEdit
		fixFiles := make(map[*ast.File]bool) // unique files involved in the current fix

		// Check the form of the declaration: var v int or struct { v int }
		def, ok := index.Def(v)
		if !ok {
			continue
		}
		var (
			typ   ast.Expr
			names []*ast.Ident
		)
		switch parent := def.Parent().Node().(type) {
		case *ast.Field: // struct { v int }
			names = parent.Names
			typ = parent.Type
		case *ast.ValueSpec: // var v int
			if len(parent.Values) > 0 {
				// e.g. var v int = 5
				// skip because rewriting as `var v atomic.Int32 = 5` is invalid
				continue
			}
			names = parent.Names
			typ = parent.Type
		}
		if len(names) != 1 || typ == nil {
			continue // v is not the sole var declared here (e.g. var x, y int32); or no explicit type
		}
		oldType := info.TypeOf(typ)                             // e.g. "int32"
		newType := strings.Title(oldType.Underlying().String()) // e.g. "Int32"

		// Get package prefix to avoid shadowing.
		file := astutil.EnclosingFile(def)
		pkgPrefix, impEdits := refactor.AddImport(pass.TypesInfo, file, "atomic", "sync/atomic", "", def.Node().Pos())
		if len(impEdits) > 0 {
			panic("unexpected import edits") // atomic PkgName should be in scope already
		}
		// Edit the type.
		//
		// var v int32
		//       ------------
		// var v atomic.Int32
		edits = append(edits, analysis.TextEdit{
			Pos:     typ.Pos(),
			End:     typ.End(),
			NewText: fmt.Appendf(nil, "%s%s", pkgPrefix, newType),
		})
		fixFiles[file] = true

		// Each valid use is an Ident v or Selector expr.v within an atomic.F(&...) call.
		var validUses []inspector.Cursor
		for cur := range index.Uses(v) {
			if v.IsField() && cur.ParentEdgeKind() == edge.KeyValueExpr_Key {
				continue nextvar // we cannot fix initial an value assignment T{v: 1}
			}
			if cur.ParentEdgeKind() == edge.SelectorExpr_Sel {
				cur = cur.Parent() // ascend from v to expr.v
			}
			// Inv: cur is the l-value expression denoting v.
			// v must appear beneath atomic.AddInt32(&v, ...) call.
			valid := false
			if cur.ParentEdgeKind() == edge.UnaryExpr_X &&
				cur.Parent().Node().(*ast.UnaryExpr).Op == token.AND {
				if ek, idx := cur.Parent().ParentEdge(); ek == edge.CallExpr_Args && idx == 0 {
					curCall := cur.Parent().Parent()
					call := curCall.Node().(*ast.CallExpr)
					if fn, ok := typeutil.Callee(info, call).(*types.Func); ok && fn.Pkg() == atomicPkg {
						valid = true
					}
				}
			}

			if !valid {
				// More complex case: reject candidate.
				//
				// For example, cur may be an unsynchronized load (e.g. v == 0). To
				// avoid a type conversion error, we'd have to rewrite this as
				// v.Load(). However, this is an invalid rewrite: if the program is
				// mixing atomic operations with unsynchronized reads, the author
				// might have accidentally introduced a data race and the suggested
				// fix could obscure the mistake. Or, if the usage is intentional,
				// rewriting may result in a behavior change.
				continue nextvar
			}
			validUses = append(validUses, cur)
		}

		for _, cur := range validUses {
			vexpr := cur.Node()
			call := cur.Parent().Parent().Node().(*ast.CallExpr)
			fn := typeutil.Callee(info, call).(*types.Func)
			// atomic.AddInt32(&v,    ...)
			// ----------------- -----
			//                  v.Add(...)
			after := vexpr.End() // LoadInt32(&v⁁)
			if len(call.Args) > 1 {
				after = call.Args[1].Pos() // AddInt32(&v, ⁁...)
			}
			verb := strings.TrimSuffix(fn.Name(), newType) // "AddInt32" => "Add"
			edits = append(edits, []analysis.TextEdit{
				{
					Pos: call.Pos(),
					End: vexpr.Pos(),
				},
				{
					Pos:     vexpr.End(),
					End:     after,
					NewText: fmt.Appendf(nil, ".%s(", verb),
				},
			}...)
			fixFiles[astutil.EnclosingFile(cur)] = true
		}

		// Check minimum Go version: go1.19, or 1.23 for the And/Or functions.
		if !(analyzerutil.FileUsesGoVersion(pass, file, versions.Go1_19) ||
			analyzerutil.FileUsesGoVersion(pass, file, versions.Go1_23) &&
				(strings.HasPrefix(funcName, "And") || strings.HasPrefix(funcName, "Or"))) {
			continue
		}

		// Skip if v is not local and the package has ignored files as it may be
		// an incomplete transformation.
		if !isLocal(v) && len(pass.IgnoredFiles) > 0 {
			continue
		}

		pass.Report(analysis.Diagnostic{
			Pos:     names[0].Pos(),
			End:     typ.End(),
			Message: fmt.Sprintf("var %s %s may be simplified using atomic.%s", v.Name(), oldType, newType),
			SuggestedFixes: []analysis.SuggestedFix{{
				Message:   fmt.Sprintf("Replace %s by atomic.%s", oldType, newType),
				TextEdits: edits,
			}},
		})
	}

	return nil, nil
}
