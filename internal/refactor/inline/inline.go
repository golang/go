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
// There are many aspects to a function call. It is the only construct
// that can simultaneously bind multiple variables of different
// explicit types, with implicit assignment conversions. (Neither var
// nor := declarations can do that.) It defines the scope of control
// labels, of return statements, and of defer statements. Arguments
// and results of function calls may be tuples even though tuples are
// not first-class values in Go, and a tuple-valued call expression
// may be "spread" across the argument list of a call or the operands
// of a return statement. All these unique features mean that in the
// general case, not everything that can be expressed by a function
// call can be expressed without one.
//
// So, in general, inlining consists of modifying a function or method
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
// In many cases it is possible to entirely replace ("reduce") the
// call by a copy of the function's body in which parameters have been
// replaced by arguments. The inliner supports a number of reduction
// strategies, and we expect this set to grow. Nonetheless, sound
// reduction is surprisingly tricky.
//
// The inliner is in some ways like an optimizing compiler. A compiler
// is considered correct if it doesn't change the meaning of the
// program in translation from source language to target language. An
// optimizing compiler exploits the particulars of the input to
// generate better code, where "better" usually means more efficient.
// When a case is found in which it emits suboptimal code, the
// compiler is improved to recognize more cases, or more rules, and
// more exceptions to rules; this process has no end. Inlining is
// similar except that "better" code means tidier code. The baseline
// translation (literalization) is correct, but there are endless
// rules--and exceptions to rules--by which the output can be
// improved.
//
// The following section lists some of the challenges, and ways in
// which they can be addressed.
//
//   - All effects of the call argument expressions must be preserved,
//     both in their number (they must not be eliminated or repeated),
//     and in their order (both with respect to other arguments, and any
//     effects in the callee function).
//
//     This must be the case even if the corresponding parameters are
//     never referenced, are referenced multiple times, referenced in
//     a different order from the arguments, or referenced within a
//     nested function that may be executed an arbitrary number of
//     times.
//
//     Currently, parameter replacement is not applied to arguments
//     with effects, but with further analysis of the sequence of
//     strict effects within the callee we could relax this constraint.
//
//   - Even an argument expression as simple as ptr.x may not be
//     referentially transparent, because another argument may have the
//     effect of changing the value of ptr.
//
//     This constraint could be relaxed by some kind of alias or
//     escape analysis that proves that ptr cannot be mutated during
//     the call.
//
//   - Although constants are referentially transparent, as a matter of
//     style we do not wish to duplicate literals that are referenced
//     multiple times in the body because this undoes proper factoring.
//     Also, string literals may be arbitrarily large.
//
//   - If the function body consists of statements other than just
//     "return expr", in some contexts it may be syntactically
//     impossible to reduce the call. Consider:
//
//     } else if x := f(); cond { ... }
//
//     Go has no equivalent to Lisp's progn or Rust's blocks,
//     nor ML's let expressions (let param = arg in body);
//     its closest equivalent is func(param){body}(arg).
//     Reduction strategies must therefore consider the syntactic
//     context of the call.
//
//   - Similarly, without the equivalent of Rust-style blocks and
//     first-class tuples, there is no general way to reduce a call
//     to a function such as
//
//     func(params)(args)(results) { stmts; return expr }
//
//     to an expression such as
//
//     { var params = args; stmts; expr }
//
//     or even a statement such as
//
//     results = { var params = args; stmts; expr }
//
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
//
//     func f(a int) int { a++; return a }
//
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
//   - Because the scope of a control label is the entire function, a
//     call cannot be reduced if the caller and callee have intersecting
//     sets of control labels. (It is possible to α-rename any
//     conflicting ones, but our colleagues building C++ refactoring
//     tools report that, when tools must choose new identifiers, they
//     generally do a poor job.)
//
//   - Given
//
//     func f() uint8 { return 0 }
//
//     var x any = f()
//
//     reducing the call to var x any = 0 is unsound because it
//     discards the implicit conversion to uint8. We may need to make
//     each argument-to-parameter conversion explicit if the types
//     differ. Assignments to variadic parameters may need to
//     explicitly construct a slice.
//
//     An analogous problem applies to the implicit assignments in
//     return statements:
//
//     func g() any { return f() }
//
//     Replacing the call f() with 0 would silently lose a
//     conversion to uint8 and change the behavior of the program.
//
//   - When inlining a call f(1, x, g()) where those parameters are
//     unreferenced, we should be able to avoid evaluating 1 and x
//     since they are pure and thus have no effect. But x may be the
//     last reference to a local variable in the caller, so removing
//     it would cause a compilation error. Argument elimination must
//     avoid making the caller's local variables unreferenced (or must
//     be prepared to eliminate the declaration too---this is where an
//     iterative framework for simplification would really help).
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
//   - Compute precisely (not conservatively) when parameter
//     elimination would remove the last reference to a caller local
//     variable, and blank out the local instead of retreating from
//     the elimination.
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
//
//     { var x int = 1; var y int32 = 2; body ...}
//
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
//
//     f := func(...) { ... }
//
//     f()
//
//     including recursive ones:
//
//     var f func(...)
//
//     f = func(...) { ...f...}
//
//     f()
//
//     But note that the existing algorithm makes widespread assumptions
//     that the callee is a package-level function or method.
//
//   - Eliminate parens inserted conservatively when they are redundant.
//
//   - Allow non-'go' build systems such as Bazel/Blaze a chance to
//     decide whether an import is accessible using logic other than
//     "/internal/" path segments. This could be achieved by returning
//     the list of added import paths instead of a text diff.
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
	"go/format"
	"go/parser"
	"go/token"
	"go/types"
	pathpkg "path"
	"reflect"
	"strconv"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/imports"
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

// Inline inlines the called function (callee) into the function call (caller)
// and returns the updated, formatted content of the caller source file.
//
// Inline does not mutate any part of Caller or Callee.
//
// The caller may supply a log function to observe the decision-making process.
//
// TODO(adonovan): provide an API for clients that want structured
// output: a list of import additions and deletions plus one or more
// localized diffs (or even AST transformations, though ownership and
// mutation are tricky) near the call site.
func Inline(logf func(string, ...any), caller *Caller, callee *Callee) ([]byte, error) {
	if logf == nil {
		logf = func(string, ...any) {} // discard
	}
	logf("inline %s @ %v",
		debugFormatNode(caller.Fset, caller.Call),
		caller.Fset.Position(caller.Call.Lparen))

	res, err := inline(logf, caller, &callee.impl)
	if err != nil {
		return nil, err
	}

	// Replace the call (or some node that encloses it) by new syntax.
	assert(res.old != nil, "old is nil")
	assert(res.new != nil, "new is nil")

	// Don't call replaceNode(caller.File, res.old, res.new)
	// as it mutates the caller's syntax tree.
	// Instead, splice the file, replacing the extent of the "old"
	// node by a formatting of the "new" node, and re-parse.
	// We'll fix up the imports on this new tree, and format again.
	var f *ast.File
	{
		start := offsetOf(caller.Fset, res.old.Pos())
		end := offsetOf(caller.Fset, res.old.End())
		var out bytes.Buffer
		out.Write(caller.Content[:start])
		// TODO(adonovan): might it make more sense to use
		// callee.Fset when formatting res.new??
		if err := format.Node(&out, caller.Fset, res.new); err != nil {
			return nil, err
		}
		out.Write(caller.Content[end:])
		const mode = parser.ParseComments | parser.SkipObjectResolution | parser.AllErrors
		f, err = parser.ParseFile(caller.Fset, "callee.go", &out, mode)
		if err != nil {
			// Something has gone very wrong.
			logf("failed to parse <<%s>>", &out) // debugging
			return nil, err
		}
	}

	// Add new imports.
	//
	// Insert new imports after last existing import,
	// to avoid migration of pre-import comments.
	// The imports will be organized below.
	if len(res.newImports) > 0 {
		var importDecl *ast.GenDecl
		if len(f.Imports) > 0 {
			// Append specs to existing import decl
			importDecl = f.Decls[0].(*ast.GenDecl)
		} else {
			// Insert new import decl.
			importDecl = &ast.GenDecl{Tok: token.IMPORT}
			f.Decls = prepend[ast.Decl](importDecl, f.Decls...)
		}
		for _, spec := range res.newImports {
			// Check that all imports (in particular, the new ones) are accessible.
			// TODO(adonovan): allow customization of the accessibility relation
			// (e.g. for Bazel).
			path, _ := strconv.Unquote(spec.Path.Value)
			// TODO(adonovan): better segment hygiene.
			if i := strings.Index(path, "/internal/"); i >= 0 {
				if !strings.HasPrefix(caller.Types.Path(), path[:i]) {
					return nil, fmt.Errorf("can't inline function %v as its body refers to inaccessible package %q", callee, path)
				}
			}
			importDecl.Specs = append(importDecl.Specs, spec)
		}
	}

	var out bytes.Buffer
	if err := format.Node(&out, caller.Fset, f); err != nil {
		return nil, err
	}
	newSrc := out.Bytes()

	// Remove imports that are no longer referenced.
	//
	// It ought to be possible to compute the set of PkgNames used
	// by the "old" code, compute the free identifiers of the
	// "new" code using a syntax-only (no go/types) algorithm, and
	// see if the reduction in the number of uses of any PkgName
	// equals the number of times it appears in caller.Info.Uses,
	// indicating that it is no longer referenced by res.new.
	//
	// However, the notorious ambiguity of resolving T{F: 0} makes this
	// unreliable: without types, we can't tell whether F refers to
	// a field of struct T, or a package-level const/var of a
	// dot-imported (!) package.
	//
	// So, for now, we run imports.Process, which is
	// unsatisfactory as it has to run the go command, and it
	// looks at the user's module cache state--unnecessarily,
	// since this step cannot add new imports.
	//
	// TODO(adonovan): replace with a simpler implementation since
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
	//
	// We could invoke imports.Process and parse its result,
	// compare against the original AST, compute a list of import
	// fixes, and return that too.

	// Recompute imports only if there were existing ones.
	if len(f.Imports) > 0 {
		formatted, err := imports.Process("output", newSrc, nil)
		if err != nil {
			logf("cannot reformat: %v <<%s>>", err, &out)
			return nil, err // cannot reformat (a bug?)
		}
		newSrc = formatted
	}
	return newSrc, nil
}

type result struct {
	newImports []*ast.ImportSpec
	old, new   ast.Node // e.g. replace call expr by callee function body expression
}

// inline returns a pair of an old node (the call, or something
// enclosing it) and a new node (its replacement, which may be a
// combination of caller, callee, and new nodes), along with the set
// of new imports needed.
//
// TODO(adonovan): rethink the 'result' interface. The assumption of a
// one-to-one replacement seems fragile. One can easily imagine the
// transformation replacing the call and adding new variable
// declarations, for example, or replacing a call statement by zero or
// many statements.)
//
// TODO(adonovan): in earlier drafts, the transformation was expressed
// by splicing substrings of the two source files because syntax
// trees don't preserve comments faithfully (see #20744), but such
// transformations don't compose. The current implementation is
// tree-based but is very lossy wrt comments. It would make a good
// candidate for evaluating an alternative fully self-contained tree
// representation, such as any proposed solution to #20744, or even
// dst or some private fork of go/ast.)
func inline(logf func(string, ...any), caller *Caller, callee *gobCallee) (*result, error) {
	checkInfoFields(caller.Info)

	// Inlining of dynamic calls is not currently supported,
	// even for local closure calls. (This would be a lot of work.)
	calleeSymbol := typeutil.StaticCallee(caller.Info, caller.Call)
	if calleeSymbol == nil {
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
	callerLookup := func(name string) types.Object {
		pos := caller.Call.Pos()
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
	var newImports []*ast.ImportSpec
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
			for n := 0; callerLookup(name) != nil; n++ {
				name = fmt.Sprintf("%s%d", base, n)
			}

			// TODO(adonovan): don't use a renaming import
			// unless the local name differs from either
			// the package name or the last segment of path.
			// This requires that we tabulate (path, declared name, local name)
			// triples for each package referenced by the callee.
			newImports = append(newImports, &ast.ImportSpec{
				Name: makeIdent(name),
				Path: &ast.BasicLit{
					Kind:  token.STRING,
					Value: strconv.Quote(path),
				},
			})
			importMap[path] = name
		}
		return name
	}

	// Compute the renaming of the callee's free identifiers.
	objRenames := make([]ast.Expr, len(callee.FreeObjs)) // nil => no change
	for i, obj := range callee.FreeObjs {
		// obj is a free object of the callee.
		//
		// Possible cases are:
		// - builtin function, type, or value (e.g. nil, zero)
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

		var newName ast.Expr
		if obj.Kind == "pkgname" {
			// Use locally appropriate import, creating as needed.
			newName = makeIdent(localImportName(obj.PkgPath)) // imported package

		} else if !obj.ValidPos {
			// Built-in function, type, or value (e.g. nil, zero):
			// check not shadowed at caller.
			found := callerLookup(obj.Name) // always finds something
			if found.Pos().IsValid() {
				return nil, fmt.Errorf("cannot inline because built-in %q is shadowed in caller by a %s (line %d)",
					obj.Name, objectKind(found),
					caller.Fset.Position(found.Pos()).Line)
			}

		} else {
			// Must be reference to package-level var/func/const/type,
			// since type parameters are not yet supported.
			qualify := false
			if obj.PkgPath == callee.PkgPath {
				// reference within callee package
				if samePkg {
					// Caller and callee are in same package.
					// Check caller has not shadowed the decl.
					found := callerLookup(obj.Name) // can't fail
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
				newName = &ast.SelectorExpr{
					X:   makeIdent(pkgName),
					Sel: makeIdent(obj.Name),
				}
			}
		}
		objRenames[i] = newName
	}

	res := &result{
		newImports: newImports,
	}

	// Parse callee function declaration.
	calleeFset, calleeDecl, err := parseCompact(callee.Content)
	if err != nil {
		return nil, err // "can't happen"
	}

	// replaceCalleeID replaces an identifier in the callee.
	// The replacement tree must not belong to the caller; use cloneNode as needed.
	replaceCalleeID := func(offset int, repl ast.Expr) {
		id := findIdent(calleeDecl, calleeDecl.Pos()+token.Pos(offset))
		logf("- replace id %q @ #%d to %q", id.Name, offset, debugFormatNode(calleeFset, repl))
		replaceNode(calleeDecl, id, repl)
	}

	// Generate replacements for each free identifier.
	// (The same tree may be spliced in multiple times, resulting in a DAG.)
	for _, ref := range callee.FreeRefs {
		if repl := objRenames[ref.Object]; repl != nil {
			replaceCalleeID(ref.Offset, repl)
		}
	}

	// Gather effective argument tuple, including receiver.
	//
	// If the receiver argument and parameter have
	// different pointerness, make the "&" or "*" explicit.
	//
	// Beware that:
	//
	// - a method can only be called through a selection, but only
	//   the first of these two forms needs special treatment:
	//
	//   expr.f(args)     -> ([&*]expr, args)	MethodVal
	//   T.f(recv, args)  -> (    expr, args)	MethodExpr
	//
	// - the presence of a value in receiver-position in the call
	//   is a property of the caller, not the callee. A method
	//   (calleeDecl.Recv != nil) may be called like an ordinary
	//   function.
	//
	// - the types.Signatures seen by the caller (from
	//   StaticCallee) and by the callee (from decl type)
	//   differ in this case.
	//
	// In a spread call f(g()), the sole ordinary argument g(),
	// always last in args, has a tuple type.
	//
	// We compute type-based predicates like pure, duplicable,
	// freevars, etc, now, before we start modifying things.
	type argument struct {
		expr       ast.Expr
		typ        types.Type      // may be tuple for sole non-receiver arg in spread call
		spread     bool            // final arg is call() assigned to multiple params
		pure       bool            // expr has no effects
		duplicable bool            // expr may be duplicated
		freevars   map[string]bool // free names of expr
	}
	var args []*argument // effective arguments; nil => eliminated
	if calleeDecl.Recv != nil {
		sel := astutil.Unparen(caller.Call.Fun).(*ast.SelectorExpr)
		if caller.Info.Selections[sel].Kind() == types.MethodVal {
			// Move receiver argument recv.f(args) to argument list f(&recv, args).
			arg := &argument{
				expr:       sel.X,
				typ:        caller.Info.TypeOf(sel.X),
				pure:       pure(caller.Info, sel.X),
				duplicable: duplicable(caller.Info, sel.X),
				freevars:   freevars(caller.Info, sel.X),
			}
			args = append(args, arg)

			// Make * or & explicit.
			//
			// We do this after we've computed the type-based
			// predicates (pure et al) above, as they won't
			// work on synthetic syntax.
			argIsPtr := arg.typ != deref(arg.typ)
			paramIsPtr := is[*types.Pointer](calleeSymbol.Type().(*types.Signature).Recv().Type())
			if !argIsPtr && paramIsPtr {
				// &recv
				arg.expr = &ast.UnaryExpr{Op: token.AND, X: arg.expr}
				arg.typ = types.NewPointer(arg.typ)
			} else if argIsPtr && !paramIsPtr {
				// *recv
				arg.expr = &ast.StarExpr{X: arg.expr}
				arg.typ = deref(arg.typ)

				// Technically *recv is non-pure and
				// non-duplicable, as side effects
				// could change the pointer between
				// multiple reads. But unfortunately
				// this really degrades many of our tests.
				//
				// TODO(adonovan): improve the precision
				// purity and duplicability.
				// For example, *new(T) is actually pure.
				// And *ptr, where ptr doesn't escape and
				// has no assignments other than its decl,
				// is also pure; this is very common.
				//
				// arg.pure = false
				// arg.duplicable = false
			}
		}
	}
	for _, expr := range caller.Call.Args {
		typ := caller.Info.TypeOf(expr)
		args = append(args, &argument{
			expr:       expr,
			typ:        typ,
			spread:     is[*types.Tuple](typ), // => last
			pure:       pure(caller.Info, expr),
			duplicable: duplicable(caller.Info, expr),
			freevars:   freevars(caller.Info, expr),
		})
	}

	// Gather effective parameter tuple, including the receiver if any.
	// Simplify variadic parameters to slices (in all cases but one).
	type parameter struct {
		obj      *types.Var // parameter var from caller's signature
		info     *paramInfo // information from AnalyzeCallee
		variadic bool       // (final) parameter is unsimplified ...T
	}
	var params []*parameter // including receiver; nil => parameter eliminated
	{
		sig := calleeSymbol.Type().(*types.Signature)
		if sig.Recv() != nil {
			params = append(params, &parameter{
				obj:  sig.Recv(),
				info: callee.Params[0],
			})
		}
		for i := 0; i < sig.Params().Len(); i++ {
			params = append(params, &parameter{
				obj:  sig.Params().At(i),
				info: callee.Params[len(params)],
			})
		}

		// Variadic function?
		//
		// There are three possible types of call:
		// - ordinary f(a1, ..., aN)
		// - ellipsis f(a1, ..., slice...)
		// - spread   f(recv?, g()) where g() is a tuple.
		// The first two are desugared to non-variadic calls
		// with an ordinary slice parameter;
		// the third is tricky and cannot be reduced, and (if
		// a receiver is present) cannot even be literalized.
		// Fortunately it is vanishingly rare.
		if sig.Variadic() {
			lastParam := last(params)
			if len(args) > 0 && last(args).spread {
				// spread call to variadic: tricky
				lastParam.variadic = true
			} else {
				// ordinary/ellipsis call to variadic

				// simplify decl: func(T...) -> func([]T)
				lastParamField := last(calleeDecl.Type.Params.List)
				lastParamField.Type = &ast.ArrayType{
					Elt: lastParamField.Type.(*ast.Ellipsis).Elt,
				}

				if caller.Call.Ellipsis.IsValid() {
					// ellipsis call: f(slice...) -> f(slice)
					// nop
				} else {
					// ordinary call: f(a1, ... aN) -> f([]T{a1, ..., aN})
					n := len(params) - 1
					ordinary, extra := args[:n], args[n:]
					var elts []ast.Expr
					pure := true
					for _, arg := range extra {
						elts = append(elts, arg.expr)
						pure = pure && arg.pure
					}
					args = append(ordinary, &argument{
						expr: &ast.CompositeLit{
							Type: lastParamField.Type,
							Elts: elts,
						},
						typ:        lastParam.obj.Type(),
						pure:       pure,
						duplicable: false,
						freevars:   nil, // not needed
					})
				}
			}
		}
	}

	// Note: computation below should be expressed in terms of
	// the args and params slices, not the raw material.

	// Parameter elimination
	//
	// Consider each parameter and its corresponding argument in turn
	// and evaluate these conditions:
	//
	// - the parameter is neither address-taken nor assigned;
	// - the argument is pure;
	// - if the parameter refcount is zero, the argument must
	//   not contain the last use of a local var;
	// - if the parameter refcount is > 1, the argument must be duplicable;
	// - the argument (or types.Default(argument) if it's untyped) has
	//   the same type as the parameter.
	//
	// If all conditions are met then the parameter can be eliminated
	// and each reference to it replaced by the argument.
	{
		// Inv:
		//  in        calls to     variadic, len(args) >= len(params)-1
		//  in spread calls to non-variadic, len(args) <  len(params)
		//  in spread calls to     variadic, len(args) <= len(params)
		// (In spread calls len(args) = 1, or 2 if call has receiver.)
		// Non-spread variadics have been simplified away already,
		// so the args[i] lookup is safe if we stop after the spread arg.
	next:
		for i, param := range params {
			arg := args[i]
			if arg.spread {
				logf("keeping param %q and following ones: argument %s is spread",
					param.info.Name, debugFormatNode(caller.Fset, arg.expr))
				break // spread => last argument, but not always last parameter
			}
			assert(!param.variadic, "unsimplfied variadic parameter")
			if param.info.Escapes {
				logf("keeping param %q: escapes from callee", param.info.Name)
				continue
			}
			if param.info.Assigned {
				logf("keeping param %q: assigned by callee", param.info.Name)
				continue // callee needs the parameter variable
			}

			// Check argument against parameter.
			//
			// Beware: don't use types.Info on arg since
			// the syntax may be synthetic (not created by parser)
			// and thus lacking positions and types;
			// do it earlier (see pure/duplicable/freevars).
			if !arg.pure {
				logf("keeping param %q: argument %s is impure",
					param.info.Name, debugFormatNode(caller.Fset, arg.expr))
				continue // unsafe to change order or cardinality of effects
			}
			if len(param.info.Refs) > 1 && !arg.duplicable {
				logf("keeping param %q: argument is not duplicable", param.info.Name)
				continue // incorrect or poor style to duplicate an expression
			}
			if len(param.info.Refs) == 0 {
				// Eliminating an unreferenced parameter might
				// remove the last reference to a caller local var.
				for free := range arg.freevars {
					if v, ok := callerLookup(free).(*types.Var); ok {
						// TODO(adonovan): be more precise and check
						// that v is defined within the body of the caller
						// function (if any) and is indeed referenced
						// only by the call.
						logf("keeping param %q: arg contains perhaps the last reference to possible caller local %v @ %v",
							param.info.Name, v, caller.Fset.Position(v.Pos()))
						continue next
					}
				}
			}

			// Check that eliminating the parameter wouldn't materially
			// change the type.
			//
			// (We don't simply wrap the argument in an explicit conversion
			// to the parameter type because that could increase allocation
			// in the number of (e.g.) string -> any conversions.
			// Even when Uses = 1, the sole ref might be in a loop or lambda that
			// is multiply executed.)
			if len(param.info.Refs) > 0 && !trivialConversion(args[i].typ, params[i].obj) {
				logf("keeping param %q: argument passing converts %s to type %s",
					param.info.Name, args[i].typ, params[i].obj.Type())
				continue // implicit conversion is significant
			}

			// Check for shadowing.
			//
			// Consider inlining a call f(z, 1) to
			// func f(x, y int) int { z := y; return x + y + z }:
			// we can't replace x in the body by z (or any
			// expression that has z as a free identifier)
			// because there's an intervening declaration of z
			// that would shadow the caller's one.
			for free := range arg.freevars {
				if param.info.Shadow[free] {
					logf("keeping param %q: cannot replace with argument as it has free ref to %s that is shadowed", param.info.Name, free)
					continue next // shadowing conflict
				}
			}

			// It is safe to eliminate param and replace it with arg.
			// No additional parens are required around arg for
			// the supported "pure" expressions.
			//
			// Because arg.expr belongs to the caller,
			// we clone it before splicing it into the callee tree.
			logf("replacing parameter %q by argument %q",
				param.info.Name, debugFormatNode(caller.Fset, arg.expr))
			for _, ref := range param.info.Refs {
				replaceCalleeID(ref, cloneNode(arg.expr).(ast.Expr))
			}
			params[i] = nil // eliminated
			args[i] = nil   // eliminated
		}
	}

	var remainingArgs []ast.Expr
	for _, arg := range args {
		if arg != nil {
			remainingArgs = append(remainingArgs, arg.expr)
		}
	}

	// TODO(adonovan): eliminate all remaining parameters
	// by replacing a call f(a1, a2)
	// to func f(x T1, y T2) {body} by
	//    { var x T1 = a1
	//      var y T2 = a2
	//      body }
	// if x ∉ freevars(a2) or freevars(T2), and so on,
	// plus the usual checks for return conversions (if any),
	// complex control, etc.
	//
	// If viable, use this with the reduction strategies below
	// that produce a block (not a value).

	// -- let the inlining strategies begin --

	// TODO(adonovan): split this huge function into a sequence of
	// function calls with an error sentinel that means "try the
	// next strategy", and make sure each strategy writes to the
	// log the reason it didn't match.

	// Special case: eliminate a call to a function whose body is empty.
	// (=> callee has no results and caller is a statement.)
	//
	//    func f(params) {}
	//    f(args)
	//    => _, _ = args
	//
	if len(calleeDecl.Body.List) == 0 {
		logf("strategy: reduce call to empty body")

		// Evaluate the arguments for effects and delete the call entirely.
		stmt := callStmt(callerPath) // cannot fail
		res.old = stmt
		if nargs := len(remainingArgs); nargs > 0 {
			// Emit "_, _ = args" to discard results.
			// Make correction for spread calls
			// f(g()) or x.f(g()) where g() is a tuple.
			//
			// TODO(adonovan): fix: it's not valid for a
			// single AssignStmt to discard a receiver and
			// a spread argument; use a var decl with two specs.
			//
			// TODO(adonovan): if args is the []T{a1, ..., an}
			// literal synthesized during variadic simplification,
			// consider unwrapping it to its (pure) elements.
			// Perhaps there's no harm doing this for any slice literal.
			if last := last(args); last != nil {
				if tuple, ok := last.typ.(*types.Tuple); ok {
					nargs += tuple.Len() - 1
				}
			}
			res.new = &ast.AssignStmt{
				Lhs: blanks[ast.Expr](nargs),
				Tok: token.ASSIGN,
				Rhs: remainingArgs,
			}
		} else {
			// No remaining arguments: delete call statement entirely
			res.new = &ast.EmptyStmt{}
		}
		return res, nil
	}

	// Attempt to reduce parameterless calls
	// whose result variables do not escape.
	allParamsEliminated := forall(params, func(i int, p *parameter) bool {
		return p == nil
	})
	noResultEscapes := !exists(callee.Results, func(i int, r *paramInfo) bool {
		return r.Escapes
	})
	if allParamsEliminated && noResultEscapes {
		logf("all params eliminated and no result vars escape")

		// Special case: parameterless call to { return exprs }.
		//
		// => reduce to:  exprs			(if legal)
		//           or:  _, _ = expr		(otherwise)
		//
		// If:
		// - the body is just "return expr" with trivial implicit conversions,
		// - all parameters have been eliminated, and
		// - no result var escapes,
		// then the call expression can be replaced by the
		// callee's body expression, suitably substituted.
		if callee.BodyIsReturnExpr {
			logf("strategy: reduce parameterless call to { return expr }")

			results := calleeDecl.Body.List[0].(*ast.ReturnStmt).Results

			clearPositions(calleeDecl.Body)

			context := callContext(callerPath)
			if stmt, ok := context.(*ast.ExprStmt); ok {
				logf("call in statement context")

				if callee.ValidForCallStmt {
					logf("callee body is valid as statement")
					// Replace the statement with the callee expr.
					res.old = caller.Call
					res.new = results[0] // Inv: len(results) == 1
				} else {
					logf("callee body is not valid as statement")
					// The call is a standalone statement, but the
					// callee body is not suitable as a standalone statement
					// (f() or <-ch), explicitly discard the results:
					// _, _ = expr
					res.old = stmt
					res.new = &ast.AssignStmt{
						Lhs: blanks[ast.Expr](callee.NumResults),
						Tok: token.ASSIGN,
						Rhs: results,
					}
				}

			} else if callee.NumResults == 1 {
				logf("call in expression context")

				// A single return operand inlined to a unary
				// expression context may need parens. Otherwise:
				//    func two() int { return 1+1 }
				//    print(-two())  =>  print(-1+1) // oops!
				//
				// TODO(adonovan): do better by analyzing 'context'
				// to see whether ambiguity is possible.
				// For example, if the context is x[y:z], then
				// the x subtree is subject to precedence ambiguity
				// (replacing x by p+q would give p+q[y:z] which is wrong)
				// but the y and z subtrees are safe.
				res.old = caller.Call
				res.new = &ast.ParenExpr{X: results[0]}

			} else {
				logf("call in spread context")

				// The call returns multiple results but is
				// not a standalone call statement. It must
				// be the RHS of a spread assignment:
				//   var x, y  = f()
				//       x, y := f()
				//       x, y  = f()
				// or the sole argument to a spread call:
				//        printf(f())
				res.old = context
				switch context := context.(type) {
				case *ast.AssignStmt:
					// Inv: the call is in Rhs[0], not Lhs.
					assign := shallowCopy(context)
					assign.Rhs = results
					res.new = assign
				case *ast.ValueSpec:
					// Inv: the call is in Values[0], not Names.
					spec := shallowCopy(context)
					spec.Values = results
					res.new = spec
				case *ast.CallExpr:
					// Inv: the Call is Args[0], not Fun.
					call := shallowCopy(context)
					call.Args = results
					res.new = call
				default:
					return nil, fmt.Errorf("internal error: unexpected context %T for spread call", context)
				}
			}
			return res, nil
		}

		// Special case: parameterless tail-call.
		//
		// Inlining:
		//         return f(args)
		// where:
		//         func f(params) (results) { body }
		// reduces to:
		//         { body }
		// so long as:
		// - all parameters are eliminated;
		// - call is a tail-call;
		// - all returns in body have trivial result conversions;
		// - there is no label conflict;
		// - no result variable is referenced by name.
		//
		// The body may use defer, arbitrary control flow, and
		// multiple returns.
		//
		// TODO(adonovan): omit the braces if the sets of
		// names in the two blocks are disjoint.
		//
		// TODO(adonovan): add a strategy for a 'void tail
		// call', i.e. a call statement prior to an (explicit
		// or implicit) return.
		if ret, ok := callContext(callerPath).(*ast.ReturnStmt); ok &&
			len(ret.Results) == 1 &&
			callee.TrivialReturns == callee.TotalReturns &&
			!hasLabelConflict(callerPath, callee.Labels) &&
			forall(callee.Results, func(i int, p *paramInfo) bool {
				// all result vars are unreferenced
				return len(p.Refs) == 0
			}) {
			logf("strategy: reduce parameterless tail-call")
			res.old = ret
			res.new = calleeDecl.Body
			clearPositions(calleeDecl.Body)
			return res, nil
		}

		// Special case: parameterless call to void function
		//
		// Inlining:
		// Special case: parameterless call to void function
		//
		// Inlining:
		//         f(args)
		// where:
		//	   func f(params) { stmts }
		// reduces to:
		//         { stmts }
		// so long as:
		// - callee is a void function (no returns)
		// - callee does not use defer
		// - there is no label conflict between caller and callee
		// - all parameters have been eliminated.
		//
		// If there is only a single statement, the braces are omitted.
		if stmt := callStmt(callerPath); stmt != nil &&
			!callee.HasDefer &&
			!hasLabelConflict(callerPath, callee.Labels) &&
			callee.TotalReturns == 0 {
			logf("strategy: reduce parameterless call to { stmt } from a call stmt")

			body := calleeDecl.Body
			var repl ast.Stmt = body
			if len(body.List) == 1 {
				repl = body.List[0] // singleton: omit braces
			}
			clearPositions(repl)
			res.old = stmt
			res.new = repl
			return res, nil
		}

		// TODO(adonovan): parameterless call to { stmt; return expr }
		// from one of these contexts:
		//    x, y     = f()
		//    x, y    := f()
		//    var x, y = f()
		// =>
		//    var (x T1, y T2); { stmts; x, y = expr }
		//
		// Because the params are no longer declared simultaneously
		// we need to check that (for example) x ∉ freevars(T2),
		// in addition to the usual checks for arg/result conversions,
		// complex control, etc.
		// Also test cases where expr is an n-ary call (spread returns).
	}

	// Literalization isn't quite infallible.
	// Consider a spread call to a method in which
	// no parameters are eliminated, e.g.
	// 	new(T).f(g())
	// where
	//  	func (recv *T) f(x, y int) { body }
	//  	func g() (int, int)
	// This would be literalized to:
	// 	func (recv *T, x, y int) { body }(new(T), g()),
	// which is not a valid argument list because g() must appear alone.
	// Reject this case for now.
	if len(args) == 2 && args[0] != nil && args[1] != nil && is[*types.Tuple](args[1].typ) {
		return nil, fmt.Errorf("can't yet inline spread call to method")
	}

	// Infallible general case: literalization.
	logf("strategy: literalization")

	// Modify callee's FuncDecl.Type.Params to remove eliminated
	// parameters and move the receiver (if any) to the head of
	// the ordinary parameters.
	//
	// The logic is fiddly because of the three forms of ast.Field:
	//   func(int), func(x int), func(x, y int)
	//
	// Also, ensure that all remaining parameters are named
	// to avoid a mix of named/unnamed when joining (recv, params...).
	// func (T) f(int, bool) -> (_ T, _ int, _ bool)
	{
		paramIdx := 0 // index in original parameter list (incl. receiver)
		var newParams []*ast.Field
		filterParams := func(field *ast.Field) {
			var names []*ast.Ident
			if field.Names == nil {
				// Unnamed parameter field (e.g. func f(int)
				if params[paramIdx] != nil {
					// Give it an explicit name "_" since we will
					// make the receiver (if any) a regular parameter
					// and one cannot mix named and unnamed parameters.
					names = blanks[*ast.Ident](1)
				}
				paramIdx++
			} else {
				// Named parameter field e.g. func f(x, y int)
				// Remove eliminated parameters in place.
				// If all were eliminated, delete field.
				for _, id := range field.Names {
					if params[paramIdx] != nil {
						names = append(names, id)
					}
					paramIdx++
				}
			}
			if names != nil {
				newParams = append(newParams, &ast.Field{
					Names: names,
					Type:  field.Type,
				})
			}
		}
		if calleeDecl.Recv != nil {
			filterParams(calleeDecl.Recv.List[0])
			calleeDecl.Recv = nil
		}
		for _, field := range calleeDecl.Type.Params.List {
			filterParams(field)
		}
		calleeDecl.Type.Params.List = newParams
	}
	// Emit a new call to a function literal in place of
	// the callee name, with appropriate replacements.
	newCall := &ast.CallExpr{
		Fun: &ast.FuncLit{
			Type: calleeDecl.Type,
			Body: calleeDecl.Body,
		},
		Ellipsis: token.NoPos, // f(slice...) is always simplified
		Args:     remainingArgs,
	}
	clearPositions(newCall.Fun)
	res.old = caller.Call
	res.new = newCall
	return res, nil
}

// -- predicates over expressions --

// freevars returns the names of all free identifiers of e:
// those lexically referenced by it but not defined within it.
// (Fields and methods are not included.)
func freevars(info *types.Info, e ast.Expr) map[string]bool {
	free := make(map[string]bool)
	ast.Inspect(e, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok {
			// The isField check is so that we don't treat T{f: 0} as a ref to f.
			if obj, ok := info.Uses[id]; ok && !within(obj.Pos(), e) && !isField(obj) {
				free[obj.Name()] = true
			}
		}
		return true
	})
	return free
}

// pure reports whether the expression is pure, that is,
// has no side effects nor potential to panic.
//
// Beware that pure does not imply referentially transparent: for
// example, new(T) is a pure expression but it returns a different
// value each time it is evaluated. (One could say that is has effects
// on the memory allocator.)
//
// TODO(adonovan):
//   - add a unit test of this function.
//   - "potential to panic": I'm not sure this is an important
//     criterion. We should be allowed to assume that good programs
//     don't rely on runtime panics for correct behavior.
//   - Should a binary + operator be considered pure? For strings, it
//     allocates memory, but so does a composite literal and that's pure
//     (but not duplicable). We need clearer definitions here.
func pure(info *types.Info, e ast.Expr) bool {
	switch e := e.(type) {
	case *ast.ParenExpr:
		return pure(info, e.X)
	case *ast.Ident:
		return true
	case *ast.FuncLit:
		return true
	case *ast.BasicLit:
		return true
	case *ast.UnaryExpr: // + - ! ^ & but not <-
		return e.Op != token.ARROW && pure(info, e.X)
	case *ast.CallExpr:
		// A conversion is considered pure
		if info.Types[e.Fun].IsType() {
			// TODO(adonovan): fix: reject the newly allowed
			// conversions between T[] and *[k]T, as they may panic.
			return pure(info, e.Args[0])
		}

		// Call to these built-ins are pure if their arguments are pure.
		if id, ok := astutil.Unparen(e.Fun).(*ast.Ident); ok {
			if b, ok := info.ObjectOf(id).(*types.Builtin); ok {
				switch b.Name() {
				case "len", "cap", "complex", "imag", "real", "make", "new", "max", "min":
					for _, arg := range e.Args {
						if !pure(info, arg) {
							return false
						}
					}
					return true
				}
			}
		}

		return false
	case *ast.KeyValueExpr:
		// map {key: value} or struct {field: value}
		return pure(info, e.Key) && pure(info, e.Value)
	case *ast.CompositeLit:
		// T{x: 0} is pure (though it may imply
		// an allocation, so it is not duplicable).
		for _, elt := range e.Elts {
			if !pure(info, elt) {
				return false
			}
		}
		return true
	case *ast.SelectorExpr:
		if sel, ok := info.Selections[e]; ok {
			// A field or method selection x.f is pure
			// if it does not indirect a pointer.
			return !sel.Indirect()
		}
		// A qualified identifier pkg.Name is pure.
		return true
	case *ast.StarExpr:
		return false // *ptr may panic
	default:
		return false
	}
}

// duplicable reports whether it is appropriate for the expression to
// be freely duplicated.
//
// Given the declaration
//
//	func f(x T) T { return x + g() + x }
//
// an argument y is considered duplicable if we would wish to see a
// call f(y) simplified to y+g()+y. This is true for identifiers,
// integer literals, unary negation, and selectors x.f where x is not
// a pointer. But we would not wish to duplicate expressions that:
// - have side effects (e.g. nearly all calls),
// - are not referentially transparent (e.g. &T{}, ptr.field), or
// - are long (e.g. "huge string literal").
func duplicable(info *types.Info, e ast.Expr) bool {
	switch e := e.(type) {
	case *ast.ParenExpr:
		return duplicable(info, e.X)
	case *ast.Ident:
		return true
	case *ast.BasicLit:
		return e.Kind == token.INT
	case *ast.UnaryExpr: // e.g. +1, -1
		return (e.Op == token.ADD || e.Op == token.SUB) && duplicable(info, e.X)
	case *ast.CallExpr:
		// Don't treat a conversion T(x) as duplicable even
		// if x is duplicable because it could duplicate
		// allocations. There may be cases to tease apart here.
		return false
	case *ast.SelectorExpr:
		if sel, ok := info.Selections[e]; ok {
			// A field or method selection x.f is referentially
			// transparent if it does not indirect a pointer.
			return !sel.Indirect()
		}
		// A qualified identifier pkg.Name is referentially transparent.
		return true
	default:
		return false
	}
}

// -- inline helpers --

func assert(cond bool, msg string) {
	if !cond {
		panic(msg)
	}
}

// blanks returns a slice of n > 0 blank identifiers.
func blanks[E ast.Expr](n int) []E {
	if n == 0 {
		panic("blanks(0)")
	}
	res := make([]E, n)
	for i := range res {
		res[i] = any(makeIdent("_")).(E) // ugh
	}
	return res
}

func makeIdent(name string) *ast.Ident {
	return &ast.Ident{Name: name}
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

// callContext returns the node immediately enclosing the call
// (specified as a PathEnclosingInterval), ignoring parens.
func callContext(callPath []ast.Node) ast.Node {
	_ = callPath[0].(*ast.CallExpr) // sanity check
	for _, n := range callPath[1:] {
		if !is[*ast.ParenExpr](n) {
			return n
		}
	}
	return nil
}

// hasLabelConflict reports whether the set of labels of the function
// enclosing the call (specified as a PathEnclosingInterval)
// intersects with the set of callee labels.
func hasLabelConflict(callPath []ast.Node, calleeLabels []string) bool {
	var callerBody *ast.BlockStmt
	switch f := callerFunc(callPath).(type) {
	case *ast.FuncDecl:
		callerBody = f.Body
	case *ast.FuncLit:
		callerBody = f.Body
	}
	conflict := false
	if callerBody != nil {
		ast.Inspect(callerBody, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.FuncLit:
				return false // prune traversal
			case *ast.LabeledStmt:
				for _, label := range calleeLabels {
					if label == n.Label.Name {
						conflict = true
					}
				}
			}
			return true
		})
	}
	return conflict
}

// callerFunc returns the innermost Func{Decl,Lit} node enclosing the
// call (specified as a PathEnclosingInterval).
func callerFunc(callPath []ast.Node) ast.Node {
	_ = callPath[0].(*ast.CallExpr) // sanity check
	for _, n := range callPath[1:] {
		if is[*ast.FuncDecl](n) || is[*ast.FuncLit](n) {
			return n
		}
	}
	return nil
}

// callStmt reports whether the function call (specified
// as a PathEnclosingInterval) appears within an ExprStmt,
// and returns it if so.
func callStmt(callPath []ast.Node) *ast.ExprStmt {
	stmt, _ := callContext(callPath).(*ast.ExprStmt)
	return stmt
}

// replaceNode performs a destructive update of the tree rooted at
// root, replacing each occurrence of "from" with "to". If to is nil and
// the element is within a slice, the slice element is removed.
//
// The root itself cannot be replaced; an attempt will panic.
//
// This function must not be called on the caller's syntax tree.
//
// TODO(adonovan): polish this up and move it to astutil package.
// TODO(adonovan): needs a unit test.
func replaceNode(root ast.Node, from, to ast.Node) {
	if from == nil {
		panic("from == nil")
	}
	if reflect.ValueOf(from).IsNil() {
		panic(fmt.Sprintf("from == (%T)(nil)", from))
	}
	if from == root {
		panic("from == root")
	}
	found := false
	var parent reflect.Value // parent variable of interface type, containing a pointer
	var visit func(reflect.Value)
	visit = func(v reflect.Value) {
		switch v.Kind() {
		case reflect.Ptr:
			if v.Interface() == from {
				found = true

				// If v is a struct field or array element
				// (e.g. Field.Comment or Field.Names[i])
				// then it is addressable (a pointer variable).
				//
				// But if it was the value an interface
				// (e.g. *ast.Ident within ast.Node)
				// then it is non-addressable, and we need
				// to set the enclosing interface (parent).
				if !v.CanAddr() {
					v = parent
				}

				// to=nil => use zero value
				var toV reflect.Value
				if to != nil {
					toV = reflect.ValueOf(to)
				} else {
					toV = reflect.Zero(v.Type()) // e.g. ast.Expr(nil)
				}
				v.Set(toV)

			} else if !v.IsNil() {
				switch v.Interface().(type) {
				case *ast.Object, *ast.Scope:
					// Skip fields of types potentially involved in cycles.
				default:
					visit(v.Elem())
				}
			}

		case reflect.Struct:
			for i := 0; i < v.Type().NumField(); i++ {
				visit(v.Field(i))
			}

		case reflect.Slice:
			compact := false
			for i := 0; i < v.Len(); i++ {
				visit(v.Index(i))
				if v.Index(i).IsNil() {
					compact = true
				}
			}
			if compact {
				// Elements were deleted. Eliminate nils.
				// (Do this is a second pass to avoid
				// unnecessary writes in the common case.)
				j := 0
				for i := 0; i < v.Len(); i++ {
					if !v.Index(i).IsNil() {
						v.Index(j).Set(v.Index(i))
						j++
					}
				}
				v.SetLen(j)
			}
		case reflect.Interface:
			parent = v
			visit(v.Elem())

		case reflect.Array, reflect.Chan, reflect.Func, reflect.Map, reflect.UnsafePointer:
			panic(v) // unreachable in AST
		default:
			// bool, string, number: nop
		}
		parent = reflect.Value{}
	}
	visit(reflect.ValueOf(root))
	if !found {
		panic(fmt.Sprintf("%T not found", from))
	}
}

// cloneNode returns a deep copy of a Node.
// It omits pointers to ast.{Scope,Object} variables.
func cloneNode(n ast.Node) ast.Node {
	var clone func(x reflect.Value) reflect.Value
	set := func(dst, src reflect.Value) {
		src = clone(src)
		if src.IsValid() {
			dst.Set(src)
		}
	}
	clone = func(x reflect.Value) reflect.Value {
		switch x.Kind() {
		case reflect.Ptr:
			if x.IsNil() {
				return x
			}
			// Skip fields of types potentially involved in cycles.
			switch x.Interface().(type) {
			case *ast.Object, *ast.Scope:
				return reflect.Zero(x.Type())
			}
			y := reflect.New(x.Type().Elem())
			set(y.Elem(), x.Elem())
			return y

		case reflect.Struct:
			y := reflect.New(x.Type()).Elem()
			for i := 0; i < x.Type().NumField(); i++ {
				set(y.Field(i), x.Field(i))
			}
			return y

		case reflect.Slice:
			y := reflect.MakeSlice(x.Type(), x.Len(), x.Cap())
			for i := 0; i < x.Len(); i++ {
				set(y.Index(i), x.Index(i))
			}
			return y

		case reflect.Interface:
			y := reflect.New(x.Type()).Elem()
			set(y, x.Elem())
			return y

		case reflect.Array, reflect.Chan, reflect.Func, reflect.Map, reflect.UnsafePointer:
			panic(x) // unreachable in AST

		default:
			return x // bool, string, number
		}
	}
	return clone(reflect.ValueOf(n)).Interface().(ast.Node)
}

// clearPositions destroys token.Pos information within the tree rooted at root,
// as positions in callee trees may cause caller comments to be emitted prematurely.
//
// In general it isn't safe to clear a valid Pos because some of them
// (e.g. CallExpr.Ellipsis, TypeSpec.Assign) are significant to
// go/printer, so this function sets each non-zero Pos to 1, which
// suffices to avoid advancing the printer's comment cursor.
//
// This function mutates its argument; do not invoke on caller syntax.
//
// TODO(adonovan): remove this horrendous workaround when #20744 is finally fixed.
func clearPositions(root ast.Node) {
	posType := reflect.TypeOf(token.NoPos)
	ast.Inspect(root, func(n ast.Node) bool {
		if n != nil {
			v := reflect.ValueOf(n).Elem() // deref the pointer to struct
			fields := v.Type().NumField()
			for i := 0; i < fields; i++ {
				f := v.Field(i)
				if f.Type() == posType {
					// Clearing Pos arbitrarily is destructive,
					// as its presence may be semantically significant
					// (e.g. CallExpr.Ellipsis, TypeSpec.Assign)
					// or affect formatting preferences (e.g. GenDecl.Lparen).
					if f.Interface() != token.NoPos {
						f.Set(reflect.ValueOf(token.Pos(1)))
					}
				}
			}
		}
		return true
	})
}

// findIdent returns the Ident beneath root that has the given pos.
func findIdent(root ast.Node, pos token.Pos) *ast.Ident {
	// TODO(adonovan): opt: skip subtrees that don't contain pos.
	var found *ast.Ident
	ast.Inspect(root, func(n ast.Node) bool {
		if found != nil {
			return false
		}
		if id, ok := n.(*ast.Ident); ok {
			if id.Pos() == pos {
				found = id
			}
		}
		return true
	})
	if found == nil {
		panic(fmt.Sprintf("findIdent %d not found", pos))
	}
	return found
}

func prepend[T any](elem T, slice ...T) []T {
	return append([]T{elem}, slice...)
}

// debugFormatNode formats a node or returns a formatting error.
// Its sloppy treatment of errors is appropriate only for logging.
func debugFormatNode(fset *token.FileSet, n ast.Node) string {
	var out strings.Builder
	if err := format.Node(&out, fset, n); err != nil {
		out.WriteString(err.Error())
	}
	return out.String()
}

func shallowCopy[T any](ptr *T) *T {
	copy := *ptr
	return &copy
}

// ∀
func forall[T any](list []T, f func(i int, x T) bool) bool {
	for i, x := range list {
		if !f(i, x) {
			return false
		}
	}
	return true
}

// ∃
func exists[T any](list []T, f func(i int, x T) bool) bool {
	for i, x := range list {
		if f(i, x) {
			return true
		}
	}
	return false
}

// last returns the last element of a slice, or zero if empty.
func last[T any](slice []T) T {
	n := len(slice)
	if n > 0 {
		return slice[n-1]
	}
	return *new(T)
}
