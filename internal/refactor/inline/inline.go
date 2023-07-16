// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package inline implements inlining of Go function calls.
//
// The client provides information about the caller and callee,
// including the source text, syntax tree, and type information, and
// the inliner returns the modified source file for the caller, or an
// error if the inlining operation is invalid (for example because the
// function body refers to names that are inaccessible to the caller).
//
// Although this interface demands more information from the client
// than might seem necessary, it enables smoother integration with
// existing batch and interactive tools that have their own ways of
// managing the processes of reading, parsing, and type-checking
// packages. In particular, this package does not assume that the
// caller and callee belong to the same token.FileSet or
// types.Importer realms.
//
// In general, inlining consists of modifying a function or method
// call expression f(a1, ..., an) so that the name of the function f
// is replaced ("literalized") by a literal copy of the function
// declaration, with free identifiers suitably modified to use the
// locally appropriate identifiers or perhaps constant argument
// values.
//
// Inlining must not change the semantics of the call. Semantics
// preservation is crucial for clients such as codebase maintenance
// tools that automatically inline all calls to designated functions
// on a large scale. Such tools must not introduce subtle behavior
// changes. (Fully inlining a call is dynamically observable using
// reflection over the call stack, but this exception to the rule is
// explicitly allowed.)
//
// In some special cases it is possible to entirely replace ("reduce")
// the call by a copy of the function's body in which parameters have
// been replaced by arguments, but this is surprisingly tricky for a
// number of reasons, some of which are listed here for illustration:
//
//   - Any effects of the call argument expressions must be preserved,
//     even if the corresponding parameters are never referenced, or are
//     referenced multiple times, or are referenced in a different order
//     from the arguments.
//
//   - Even an argument expression as simple as ptr.x may not be
//     referentially transparent, because another argument may have the
//     effect of changing the value of ptr.
//
//   - Although constants are referentially transparent, as a matter of
//     style we do not wish to duplicate literals that are referenced
//     multiple times in the body because this undoes proper factoring.
//     Also, string literals may be arbitrarily large.
//
//   - If the function body consists of statements other than just
//     "return expr", in some contexts it may be syntactically
//     impossible to replace the call expression by the body statements.
//     Consider "} else if x := f(); cond { ... }".
//     (Go has no equivalent to Lisp's progn or Rust's blocks.)
//
//   - Similarly, without the equivalent of Rust-style blocks and
//     first-class tuples, there is no general way to reduce a call
//     to a function such as
//     >	func(params)(args)(results) { stmts; return body }
//     to an expression such as
//     >	{ var params = args; stmts; body }
//     or even a statement such as
//     >	results = { var params = args; stmts; body }
//     Consequently the declaration and scope of the result variables,
//     and the assignment and control-flow implications of the return
//     statement, must be dealt with by cases.
//
//   - A standalone call statement that calls a function whose body is
//     "return expr" cannot be simply replaced by the body expression
//     if it is not itself a call or channel receive expression; it is
//     necessary to explicitly discard the result using "_ = expr".
//
//     Similarly, if the body is a call expression, only calls to some
//     built-in functions with no result (such as copy or panic) are
//     permitted as statements, whereas others (such as append) return
//     a result that must be used, even if just by discarding.
//
//   - If a parameter or result variable is updated by an assignment
//     within the function body, it cannot always be safely replaced
//     by a variable in the caller. For example, given
//     >	func f(a int) int { a++; return a }
//     The call y = f(x) cannot be replaced by { x++; y = x } because
//     this would change the value of the caller's variable x.
//     Only if the caller is finished with x is this safe.
//
//     A similar argument applies to parameter or result variables
//     that escape: by eliminating a variable, inlining would change
//     the identity of the variable that escapes.
//
//   - If the function body uses 'defer' and the inlined call is not a
//     tail-call, inlining may delay the deferred effects.
//
//   - Each control label that is used by both caller and callee must
//     be α-renamed.
//
//   - Given
//     >	func f() uint8 { return 0 }
//     >	var x any = f()
//     reducing the call to var x any = 0 is unsound because it
//     discards the implicit conversion. We may need to make each
//     argument->parameter and return->result assignment conversion
//     implicit if the types differ. Assignments to variadic
//     parameters may need to explicitly construct a slice.
//
// More complex callee functions are inlinable with more elaborate and
// invasive changes to the statements surrounding the call expression.
//
// TODO(adonovan): future work:
//
//   - Handle more of the above special cases by careful analysis,
//     thoughtful factoring of the large design space, and thorough
//     test coverage.
//
//   - Write a fuzz-like test that selects function calls at
//     random in the corpus, inlines them, and checks that the
//     result is either a sensible error or a valid transformation.
//
//   - Eliminate parameters that are unreferenced in the callee
//     and whose argument expression is side-effect free.
//
//   - Afford the client more control such as a limit on the total
//     increase in line count, or a refusal to inline using the
//     general approach (replacing name by function literal). This
//     could be achieved by returning metadata alongside the result
//     and having the client conditionally discard the change.
//
//   - Is it acceptable to skip effects that are limited to runtime
//     panics? Can we avoid evaluating an argument x.f
//     or a[i] when the corresponding parameter is unused?
//
//   - When caller syntax permits a block, replace argument-to-parameter
//     assignment by a set of local var decls, e.g. f(1, 2) would
//     become { var x, y = 1, 2; body... }.
//
//     But even this is complicated: a single var decl initializer
//     cannot declare all the parameters and initialize them to their
//     arguments in one go if they have varied types. Instead,
//     one must use multiple specs such as:
//     >	{ var x int = 1; var y int32 = 2; body ...}
//     but this means that the initializer expression for y is
//     within the scope of x, so it may require α-renaming.
//
//     It is tempting to use a short var decl { x, y := 1, 2; body ...}
//     as it permits simultaneous declaration and initialization
//     of many variables of varied type. However, one must take care
//     to convert each argument expression to the correct parameter
//     variable type, perhaps explicitly. (Consider "x := 1 << 64".)
//
//     Also, as a matter of style, having all parameter declarations
//     and argument expressions in a single statement is potentially
//     unwieldy.
//
//   - Support inlining of generic functions, replacing type parameters
//     by their instantiations.
//
//   - Support inlining of calls to function literals such as:
//     >	f := func(...) { ...}
//     >	f()
//     including recursive ones:
//     >	var f func(...)
//     >	f = func(...) { ...f...}
//     >	f()
//     But note that the existing algorithm makes widespread assumptions
//     that the callee is a package-level function or method.
//
//   - Eliminate parens inserted conservatively when they are redundant.
//
//   - Allow non-'go' build systems such as Bazel/Blaze a chance to
//     decide whether an import is accessible using logic other than
//     "/internal/" path segments. This could be achieved by returning
//     the list of added import paths.
//
//   - Inlining a function from another module may change the
//     effective version of the Go language spec that governs it. We
//     should probably make the client responsible for rejecting
//     attempts to inline from newer callees to older callers, since
//     there's no way for this package to access module versions.
//
//   - Use an alternative implementation of the import-organizing
//     operation that doesn't require operating on a complete file
//     (and reformatting). Then return the results in a higher-level
//     form as a set of import additions and deletions plus a single
//     diff that encloses the call expression. This interface could
//     perhaps be implemented atop imports.Process by post-processing
//     its result to obtain the abstract import changes and discarding
//     its formatted output.
package inline

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"log"
	pathpkg "path"
	"reflect"
	"sort"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/imports"
	"golang.org/x/tools/internal/typeparams"
)

// A Caller describes the function call and its enclosing context.
//
// The client is responsible for populating this struct and passing it to Inline.
type Caller struct {
	Fset    *token.FileSet
	Types   *types.Package
	Info    *types.Info
	File    *ast.File
	Call    *ast.CallExpr
	Content []byte
}

func (caller *Caller) offset(pos token.Pos) int { return offsetOf(caller.Fset, pos) }

// Inline inlines the called function (callee) into the function call (caller)
// and returns the updated, formatted content of the caller source file.
func Inline(caller *Caller, callee_ *Callee) ([]byte, error) {
	callee := &callee_.impl

	// -- check caller --

	// Inlining of dynamic calls is not currently supported,
	// even for local closure calls.
	if typeutil.StaticCallee(caller.Info, caller.Call) == nil {
		// e.g. interface method
		return nil, fmt.Errorf("cannot inline: not a static function call")
	}

	// Reject cross-package inlining if callee has
	// free references to unexported symbols.
	samePkg := caller.Types.Path() == callee.PkgPath
	if !samePkg && len(callee.Unexported) > 0 {
		return nil, fmt.Errorf("cannot inline call to %s because body refers to non-exported %s",
			callee.Name, callee.Unexported[0])
	}

	// -- analyze callee's free references in caller context --

	// syntax path enclosing Call, innermost first (Path[0]=Call)
	callerPath, _ := astutil.PathEnclosingInterval(caller.File, caller.Call.Pos(), caller.Call.End())
	callerLookup := func(name string, pos token.Pos) types.Object {
		for _, n := range callerPath {
			// The function body scope (containing not just params)
			// is associated with FuncDecl.Type, not FuncDecl.Body.
			if decl, ok := n.(*ast.FuncDecl); ok {
				n = decl.Type
			}
			if scope := caller.Info.Scopes[n]; scope != nil {
				if _, obj := scope.LookupParent(name, pos); obj != nil {
					return obj
				}
			}
		}
		return nil
	}

	// Import map, initially populated with caller imports.
	//
	// For simplicity we ignore existing dot imports, so that a
	// qualified identifier (QI) in the callee is always
	// represented by a QI in the caller, allowing us to treat a
	// QI like a selection on a package name.
	importMap := make(map[string]string) // maps package path to local name
	for _, imp := range caller.File.Imports {
		if pkgname, ok := importedPkgName(caller.Info, imp); ok && pkgname.Name() != "." {
			importMap[pkgname.Imported().Path()] = pkgname.Name()
		}
	}

	// localImportName returns the local name for a given imported package path.
	var newImports []string
	localImportName := func(path string) string {
		name, ok := importMap[path]
		if !ok {
			// import added by callee
			//
			// Choose local PkgName based on last segment of
			// package path plus, if needed, a numeric suffix to
			// ensure uniqueness.
			//
			// TODO(adonovan): preserve the PkgName used
			// in the original source, or, for a dot import,
			// use the package's declared name.
			base := pathpkg.Base(path)
			name = base
			for n := 0; callerLookup(name, caller.Call.Pos()) != nil; n++ {
				name = fmt.Sprintf("%s%d", base, n)
			}

			// TODO(adonovan): don't use a renaming import
			// unless the local name differs from either
			// the package name or the last segment of path.
			// This requires that we tabulate (path, declared name, local name)
			// triples for each package referenced by the callee.
			newImports = append(newImports, fmt.Sprintf("%s %q", name, path))
			importMap[path] = name
		}
		return name
	}

	// Compute the renaming of the callee's free identifiers.
	objRenames := make([]string, len(callee.FreeObjs)) // "" => no rename
	for i, obj := range callee.FreeObjs {
		// obj is a free object of the callee.
		//
		// Possible cases are:
		// - nil or a builtin
		//   => check not shadowed in caller.
		// - package-level var/func/const/types
		//   => same package: check not shadowed in caller.
		//   => otherwise: import other package form a qualified identifier.
		//      (Unexported cross-package references were rejected already.)
		// - type parameter
		//   => not yet supported
		// - pkgname
		//   => import other package and use its local name.
		//
		// There can be no free references to labels, fields, or methods.

		var newName string
		if obj.Kind == "pkgname" {
			// Use locally appropriate import, creating as needed.
			newName = localImportName(obj.PkgPath) // imported package

		} else if !obj.ValidPos {
			// Built-in function, type, or nil: check not shadowed at caller.
			found := callerLookup(obj.Name, caller.Call.Pos()) // can't fail
			if found.Pos().IsValid() {
				return nil, fmt.Errorf("cannot inline because built-in %q is shadowed in caller by a %s (line %d)",
					obj.Name, objectKind(found),
					caller.Fset.Position(found.Pos()).Line)
			}

			newName = obj.Name

		} else {
			// Must be reference to package-level var/func/const/type,
			// since type parameters are not yet supported.
			newName = obj.Name
			qualify := false
			if obj.PkgPath == callee.PkgPath {
				// reference within callee package
				if samePkg {
					// Caller and callee are in same package.
					// Check caller has not shadowed the decl.
					found := callerLookup(obj.Name, caller.Call.Pos()) // can't fail
					if !isPkgLevel(found) {
						return nil, fmt.Errorf("cannot inline because %q is shadowed in caller by a %s (line %d)",
							obj.Name, objectKind(found),
							caller.Fset.Position(found.Pos()).Line)
					}
				} else {
					// Cross-package reference.
					qualify = true
				}
			} else {
				// Reference to a package-level declaration
				// in another package, without a qualified identifier:
				// it must be a dot import.
				qualify = true
			}

			// Form a qualified identifier, pkg.Name.
			if qualify {
				pkgName := localImportName(obj.PkgPath)
				newName = pkgName + "." + newName
			}
		}
		objRenames[i] = newName
	}

	// Compute edits to inlined callee.
	type edit struct {
		start, end int // byte offsets wrt callee.content
		new        string
	}
	var edits []edit

	// Give explicit blank "_" names to all method parameters
	// (including receiver) since we will make the receiver a regular
	// parameter and one cannot mix named and unnamed parameters.
	// e.g. func (T) f(int, string) -> (_ T, _ int, _ string)
	if callee.decl.Recv != nil {
		ensureNamed := func(params *ast.FieldList) {
			for _, param := range params.List {
				if param.Names == nil {
					offset := callee.offset(param.Type.Pos())
					edits = append(edits, edit{
						start: offset,
						end:   offset,
						new:   "_ ",
					})
				}
			}
		}
		ensureNamed(callee.decl.Recv)
		ensureNamed(callee.decl.Type.Params)
	}

	// Generate replacements for each free identifier.
	for _, ref := range callee.FreeRefs {
		if repl := objRenames[ref.Object]; repl != "" {
			edits = append(edits, edit{
				start: ref.Start,
				end:   ref.End,
				new:   repl,
			})
		}
	}

	// Edits are non-overlapping but insertions and edits may be coincident.
	// Preserve original order.
	sort.SliceStable(edits, func(i, j int) bool {
		return edits[i].start < edits[j].start
	})

	// Check that all imports (in particular, the new ones) are accessible.
	// TODO(adonovan): allow customization of the accessibility relation (e.g. for Bazel).
	for path := range importMap {
		// TODO(adonovan): better segment hygiene.
		if i := strings.Index(path, "/internal/"); i >= 0 {
			if !strings.HasPrefix(caller.Types.Path(), path[:i]) {
				return nil, fmt.Errorf("can't inline function %v as its body refers to inaccessible package %q", callee.Name, path)
			}
		}
	}

	// The transformation is expressed by splicing substrings of
	// the two source files, because syntax trees don't preserve
	// comments faithfully (see #20744).
	var out bytes.Buffer

	// 'replace' emits to out the specified range of the callee,
	// applying all edits that fall completely within it.
	replace := func(start, end int) {
		off := start
		for _, edit := range edits {
			if start <= edit.start && edit.end <= end {
				out.Write(callee.Content[off:edit.start])
				out.WriteString(edit.new)
				off = edit.end
			}
		}
		out.Write(callee.Content[off:end])
	}

	// Insert new imports after last existing import,
	// to avoid migration of pre-import comments.
	// The imports will be organized later.
	{
		offset := caller.offset(caller.File.Name.End()) // after package decl
		if len(caller.File.Imports) > 0 {
			// It's tempting to insert the new import after the last ImportSpec,
			// but that may not be at the end of the import decl.
			// Consider: import ( "a"; "b" ‸ )
			for _, decl := range caller.File.Decls {
				if decl, ok := decl.(*ast.GenDecl); ok && decl.Tok == token.IMPORT {
					offset = caller.offset(decl.End()) // after import decl
				}
			}
		}
		out.Write(caller.Content[:offset])
		out.WriteString("\n")
		for _, imp := range newImports {
			fmt.Fprintf(&out, "import %s\n", imp)
		}
		out.Write(caller.Content[offset:caller.offset(caller.Call.Pos())])
	}

	// Special case: a call to a function whose body consists only
	// of "return expr" may be replaced by the expression, so long as:
	//
	// (a) There are no receiver or parameter argument expressions
	//     whose side effects must be considered.
	// (b) There are no named parameter or named result variables
	//     that could potentially escape.
	//
	// TODO(adonovan): expand this special case to cover more scenarios.
	// Consider each parameter in turn. If:
	// - the parameter does not escape and is never assigned;
	// - its argument is pure (no effects or panics--basically just idents and literals)
	//   and referentially transparent (not new(T) or &T{...}) or referenced at most once; and
	// - the argument and parameter have the same type
	// then the parameter can be eliminated and each reference
	// to it replaced by the argument.
	// If:
	// - all parameters can be so replaced;
	// - and the body is just "return expr";
	// - and the result vars are unnamed or never referenced (and thus cannot escape);
	// then the call expression can be replaced by its body expression.
	if callee.BodyIsReturnExpr &&
		callee.decl.Recv == nil && // no receiver arg effects to consider
		len(caller.Call.Args) == 0 && // no argument effects to consider
		!hasNamedVars(callee.decl.Type.Params) && // no param vars escape
		!hasNamedVars(callee.decl.Type.Results) { // no result vars escape

		// A single return operand inlined to an expression
		// context may need parens. Otherwise:
		//    func two() int { return 1+1 }
		//    print(-two())  =>  print(-1+1) // oops!
		parens := callee.NumResults == 1

		// If the call is a standalone statement, but the
		// callee body is not suitable as a standalone statement
		// (f() or <-ch), explicitly discard the results:
		// _, _ = expr
		if isCallStmt(callerPath) {
			parens = false

			if !callee.ValidForCallStmt {
				for i := 0; i < callee.NumResults; i++ {
					if i > 0 {
						out.WriteString(", ")
					}
					out.WriteString("_")
				}
				out.WriteString(" = ")
			}
		}

		// Emit the body expression(s).
		for i, res := range callee.decl.Body.List[0].(*ast.ReturnStmt).Results {
			if i > 0 {
				out.WriteString(", ")
			}
			if parens {
				out.WriteString("(")
			}
			replace(callee.offset(res.Pos()), callee.offset(res.End()))
			if parens {
				out.WriteString(")")
			}
		}
		goto rest
	}

	// Emit a function literal in place of the callee name,
	// with appropriate replacements.
	out.WriteString("func (")
	if recv := callee.decl.Recv; recv != nil {
		// Move method receiver to head of ordinary parameters.
		replace(callee.offset(recv.Opening+1), callee.offset(recv.Closing))
		if len(callee.decl.Type.Params.List) > 0 {
			out.WriteString(", ")
		}
	}
	replace(callee.offset(callee.decl.Type.Params.Opening+1),
		callee.offset(callee.decl.End()))

	// Emit call arguments.
	out.WriteString("(")
	if callee.decl.Recv != nil {
		// Move receiver argument x.f(...) to argument list f(x, ...).
		recv := astutil.Unparen(caller.Call.Fun).(*ast.SelectorExpr).X

		// If the receiver argument and parameter have
		// different pointerness, make the "&" or "*" explicit.
		argPtr := is[*types.Pointer](typeparams.CoreType(caller.Info.TypeOf(recv)))
		paramPtr := is[*ast.StarExpr](callee.decl.Recv.List[0].Type)
		if !argPtr && paramPtr {
			out.WriteString("&")
		} else if argPtr && !paramPtr {
			out.WriteString("*")
		}

		out.Write(caller.Content[caller.offset(recv.Pos()):caller.offset(recv.End())])

		if len(caller.Call.Args) > 0 {
			out.WriteString(", ")
		}
	}
	// Append ordinary args, sans initial "(".
	out.Write(caller.Content[caller.offset(caller.Call.Lparen+1):caller.offset(caller.Call.End())])

	// Append rest of caller file.
rest:
	out.Write(caller.Content[caller.offset(caller.Call.End()):])

	// Reformat, and organize imports.
	//
	// TODO(adonovan): this looks at the user's cache state.
	// Replace with a simpler implementation since
	// all the necessary imports are present but merely untidy.
	// That will be faster, and also less prone to nondeterminism
	// if there are bugs in our logic for import maintenance.
	//
	// However, golang.org/x/tools/internal/imports.ApplyFixes is
	// too simple as it requires the caller to have figured out
	// all the logical edits. In our case, we know all the new
	// imports that are needed (see newImports), each of which can
	// be specified as:
	//
	//   &imports.ImportFix{
	//     StmtInfo: imports.ImportInfo{path, name,
	//     IdentName: name,
	//     FixType:   imports.AddImport,
	//   }
	//
	// but we don't know which imports are made redundant by the
	// inlining itself. For example, inlining a call to
	// fmt.Println may make the "fmt" import redundant.
	//
	// Also, both imports.Process and internal/imports.ApplyFixes
	// reformat the entire file, which is not ideal for clients
	// such as gopls. (That said, the point of a canonical format
	// is arguably that any tool can reformat as needed without
	// this being inconvenient.)
	res, err := imports.Process("output", out.Bytes(), nil)
	if err != nil {
		if false { // debugging
			log.Printf("cannot reformat: %v <<%s>>", err, &out)
		}
		return nil, err // cannot reformat (a bug?)
	}
	return res, nil
}

// -- helpers --

func is[T any](x any) bool {
	_, ok := x.(T)
	return ok
}

func within(pos token.Pos, n ast.Node) bool {
	return n.Pos() <= pos && pos <= n.End()
}

func offsetOf(fset *token.FileSet, pos token.Pos) int {
	return fset.PositionFor(pos, false).Offset
}

// importedPkgName returns the PkgName object declared by an ImportSpec.
// TODO(adonovan): make this a method of types.Info (#62037).
func importedPkgName(info *types.Info, imp *ast.ImportSpec) (*types.PkgName, bool) {
	var obj types.Object
	if imp.Name != nil {
		obj = info.Defs[imp.Name]
	} else {
		obj = info.Implicits[imp]
	}
	pkgname, ok := obj.(*types.PkgName)
	return pkgname, ok
}

func isPkgLevel(obj types.Object) bool {
	return obj.Pkg().Scope().Lookup(obj.Name()) == obj
}

// objectKind returns an object's kind (e.g. var, func, const, typename).
func objectKind(obj types.Object) string {
	return strings.TrimPrefix(strings.ToLower(reflect.TypeOf(obj).String()), "*types.")
}

// isCallStmt reports whether the function call (specified
// as a PathEnclosingInterval) appears within an ExprStmt.
func isCallStmt(callPath []ast.Node) bool {
	_ = callPath[0].(*ast.CallExpr)
	for _, n := range callPath[1:] {
		switch n.(type) {
		case *ast.ParenExpr:
			continue
		case *ast.ExprStmt:
			return true
		}
		break
	}
	return false
}

// hasNamedVars reports whether a function parameter tuple uses named variables.
//
// TODO(adonovan): this is a placeholder for a more complex analysis to detect
// whether inlining might cause named param/result variables to escape.
func hasNamedVars(tuple *ast.FieldList) bool {
	return tuple != nil && len(tuple.List) > 0 && tuple.List[0].Names != nil
}
