// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Taken from golang.org/x/tools/refactor/rename.

package source

// This file defines the conflict-checking portion of the rename operation.
//
// The renamer works on a single package of type-checked syntax, and
// is called in parallel for all necessary packages in the workspace,
// possibly up to the transitive reverse dependencies of the
// declaration. Finally the union of all edits and errors is computed.
//
// Renaming one object may entail renaming of others. For example:
//
// - An embedded field couples a Var (field) and a TypeName.
//   So, renaming either one requires renaming the other.
//   If the initial object is an embedded field, we must add its
//   TypeName (and its enclosing package) to the renaming set;
//   this is easily discovered at the outset.
//
//   Conversely, if the initial object is a TypeName, we must observe
//   whether any of its references (from directly importing packages)
//   is coincident with an embedded field Var and, if so, initiate a
//   renaming of it.
//
// - A method of an interface type is coupled to all corresponding
//   methods of types that are assigned to the interface (as
//   discovered by the 'satisfy' pass). As a matter of usability, we
//   require that such renamings be initiated from the interface
//   method, not the concrete method.

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"path/filepath"
	"reflect"
	"strings"
	"unicode"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/refactor/satisfy"
)

// errorf reports an error (e.g. conflict) and prevents file modification.
func (r *renamer) errorf(pos token.Pos, format string, args ...interface{}) {
	// Conflict error messages in the old gorename tool (whence this
	// logic originated) contain rich information associated with
	// multiple source lines, such as:
	//
	//   p/a.go:1:2: renaming "x" to "y" here
	//   p/b.go:3:4: \t would cause this reference to "y"
	//   p/c.go:5:5: \t to become shadowed by this intervening declaration.
	//
	// Unfortunately LSP provides no means to transmit the
	// structure of this error, so we format the positions briefly
	// using dir/file.go where dir is the base name of the parent
	// directory.

	var conflict strings.Builder

	// Add prefix of (truncated) position.
	if pos != token.NoPos {
		// TODO(adonovan): skip position of first error if it is
		// on the same line as the renaming itself.
		posn := safetoken.StartPosition(r.pkg.FileSet(), pos).String()
		segments := strings.Split(filepath.ToSlash(posn), "/")
		if n := len(segments); n > 2 {
			segments = segments[n-2:]
		}
		posn = strings.Join(segments, "/")
		fmt.Fprintf(&conflict, "%s:", posn)

		if !strings.HasPrefix(format, "\t") {
			conflict.WriteByte(' ')
		}
	}

	fmt.Fprintf(&conflict, format, args...)
	r.conflicts = append(r.conflicts, conflict.String())
}

// check performs safety checks of the renaming of the 'from' object to r.to.
func (r *renamer) check(from types.Object) {
	if r.objsToUpdate[from] {
		return
	}
	r.objsToUpdate[from] = true

	// NB: order of conditions is important.
	if from_, ok := from.(*types.PkgName); ok {
		r.checkInFileBlock(from_)
	} else if from_, ok := from.(*types.Label); ok {
		r.checkLabel(from_)
	} else if isPackageLevel(from) {
		r.checkInPackageBlock(from)
	} else if v, ok := from.(*types.Var); ok && v.IsField() {
		r.checkStructField(v)
	} else if f, ok := from.(*types.Func); ok && recv(f) != nil {
		r.checkMethod(f)
	} else if isLocal(from) {
		r.checkInLexicalScope(from)
	} else {
		r.errorf(from.Pos(), "unexpected %s object %q (please report a bug)\n",
			objectKind(from), from)
	}
}

// checkInFileBlock performs safety checks for renames of objects in the file block,
// i.e. imported package names.
func (r *renamer) checkInFileBlock(from *types.PkgName) {
	// Check import name is not "init".
	if r.to == "init" {
		r.errorf(from.Pos(), "%q is not a valid imported package name", r.to)
	}

	// Check for conflicts between file and package block.
	if prev := from.Pkg().Scope().Lookup(r.to); prev != nil {
		r.errorf(from.Pos(), "renaming this %s %q to %q would conflict",
			objectKind(from), from.Name(), r.to)
		r.errorf(prev.Pos(), "\twith this package member %s",
			objectKind(prev))
		return // since checkInPackageBlock would report redundant errors
	}

	// Check for conflicts in lexical scope.
	r.checkInLexicalScope(from)
}

// checkInPackageBlock performs safety checks for renames of
// func/var/const/type objects in the package block.
func (r *renamer) checkInPackageBlock(from types.Object) {
	// Check that there are no references to the name from another
	// package if the renaming would make it unexported.
	if typ := r.pkg.GetTypes(); typ != from.Pkg() && ast.IsExported(r.from) && !ast.IsExported(r.to) {
		if id := someUse(r.pkg.GetTypesInfo(), from); id != nil {
			r.checkExport(id, typ, from)
		}
	}

	// Check that in the package block, "init" is a function, and never referenced.
	if r.to == "init" {
		kind := objectKind(from)
		if kind == "func" {
			// Reject if intra-package references to it exist.
			for id, obj := range r.pkg.GetTypesInfo().Uses {
				if obj == from {
					r.errorf(from.Pos(),
						"renaming this func %q to %q would make it a package initializer",
						from.Name(), r.to)
					r.errorf(id.Pos(), "\tbut references to it exist")
					break
				}
			}
		} else {
			r.errorf(from.Pos(), "you cannot have a %s at package level named %q",
				kind, r.to)
		}
	}

	// Check for conflicts between package block and all file blocks.
	for _, f := range r.pkg.GetSyntax() {
		fileScope := r.pkg.GetTypesInfo().Scopes[f]
		b, prev := fileScope.LookupParent(r.to, token.NoPos)
		if b == fileScope {
			r.errorf(from.Pos(), "renaming this %s %q to %q would conflict", objectKind(from), from.Name(), r.to)
			var prevPos token.Pos
			if prev != nil {
				prevPos = prev.Pos()
			}
			r.errorf(prevPos, "\twith this %s", objectKind(prev))
			return // since checkInPackageBlock would report redundant errors
		}
	}

	// Check for conflicts in lexical scope.
	r.checkInLexicalScope(from)
}

// checkInLexicalScope performs safety checks that a renaming does not
// change the lexical reference structure of the specified package.
//
// For objects in lexical scope, there are three kinds of conflicts:
// same-, sub-, and super-block conflicts.  We will illustrate all three
// using this example:
//
//	var x int
//	var z int
//
//	func f(y int) {
//		print(x)
//		print(y)
//	}
//
// Renaming x to z encounters a "same-block conflict", because an object
// with the new name already exists, defined in the same lexical block
// as the old object.
//
// Renaming x to y encounters a "sub-block conflict", because there exists
// a reference to x from within (what would become) a hole in its scope.
// The definition of y in an (inner) sub-block would cast a shadow in
// the scope of the renamed variable.
//
// Renaming y to x encounters a "super-block conflict".  This is the
// converse situation: there is an existing definition of the new name
// (x) in an (enclosing) super-block, and the renaming would create a
// hole in its scope, within which there exist references to it.  The
// new name shadows the existing definition of x in the super-block.
//
// Removing the old name (and all references to it) is always safe, and
// requires no checks.
func (r *renamer) checkInLexicalScope(from types.Object) {
	b := from.Parent() // the block defining the 'from' object
	if b != nil {
		toBlock, to := b.LookupParent(r.to, from.Parent().End())
		if toBlock == b {
			// same-block conflict
			r.errorf(from.Pos(), "renaming this %s %q to %q",
				objectKind(from), from.Name(), r.to)
			r.errorf(to.Pos(), "\tconflicts with %s in same block",
				objectKind(to))
			return
		} else if toBlock != nil {
			// Check for super-block conflict.
			// The name r.to is defined in a superblock.
			// Is that name referenced from within this block?
			forEachLexicalRef(r.pkg, to, func(id *ast.Ident, block *types.Scope) bool {
				_, obj := block.LookupParent(from.Name(), id.Pos())
				if obj == from {
					// super-block conflict
					r.errorf(from.Pos(), "renaming this %s %q to %q",
						objectKind(from), from.Name(), r.to)
					r.errorf(id.Pos(), "\twould shadow this reference")
					r.errorf(to.Pos(), "\tto the %s declared here",
						objectKind(to))
					return false // stop
				}
				return true
			})
		}
	}
	// Check for sub-block conflict.
	// Is there an intervening definition of r.to between
	// the block defining 'from' and some reference to it?
	forEachLexicalRef(r.pkg, from, func(id *ast.Ident, block *types.Scope) bool {
		// Find the block that defines the found reference.
		// It may be an ancestor.
		fromBlock, _ := block.LookupParent(from.Name(), id.Pos())
		// See what r.to would resolve to in the same scope.
		toBlock, to := block.LookupParent(r.to, id.Pos())
		if to != nil {
			// sub-block conflict
			if deeper(toBlock, fromBlock) {
				r.errorf(from.Pos(), "renaming this %s %q to %q",
					objectKind(from), from.Name(), r.to)
				r.errorf(id.Pos(), "\twould cause this reference to become shadowed")
				r.errorf(to.Pos(), "\tby this intervening %s definition",
					objectKind(to))
				return false // stop
			}
		}
		return true
	})

	// Renaming a type that is used as an embedded field
	// requires renaming the field too. e.g.
	// 	type T int // if we rename this to U..
	// 	var s struct {T}
	// 	print(s.T) // ...this must change too
	if _, ok := from.(*types.TypeName); ok {
		for id, obj := range r.pkg.GetTypesInfo().Uses {
			if obj == from {
				if field := r.pkg.GetTypesInfo().Defs[id]; field != nil {
					r.check(field)
				}
			}
		}
	}
}

// deeper reports whether block x is lexically deeper than y.
func deeper(x, y *types.Scope) bool {
	if x == y || x == nil {
		return false
	} else if y == nil {
		return true
	} else {
		return deeper(x.Parent(), y.Parent())
	}
}

// forEachLexicalRef calls fn(id, block) for each identifier id in package
// pkg that is a reference to obj in lexical scope.  block is the
// lexical block enclosing the reference.  If fn returns false the
// iteration is terminated and findLexicalRefs returns false.
func forEachLexicalRef(pkg Package, obj types.Object, fn func(id *ast.Ident, block *types.Scope) bool) bool {
	ok := true
	var stack []ast.Node

	var visit func(n ast.Node) bool
	visit = func(n ast.Node) bool {
		if n == nil {
			stack = stack[:len(stack)-1] // pop
			return false
		}
		if !ok {
			return false // bail out
		}

		stack = append(stack, n) // push
		switch n := n.(type) {
		case *ast.Ident:
			if pkg.GetTypesInfo().Uses[n] == obj {
				block := enclosingBlock(pkg.GetTypesInfo(), stack)
				if !fn(n, block) {
					ok = false
				}
			}
			return visit(nil) // pop stack

		case *ast.SelectorExpr:
			// don't visit n.Sel
			ast.Inspect(n.X, visit)
			return visit(nil) // pop stack, don't descend

		case *ast.CompositeLit:
			// Handle recursion ourselves for struct literals
			// so we don't visit field identifiers.
			tv, ok := pkg.GetTypesInfo().Types[n]
			if !ok {
				return visit(nil) // pop stack, don't descend
			}
			if _, ok := Deref(tv.Type).Underlying().(*types.Struct); ok {
				if n.Type != nil {
					ast.Inspect(n.Type, visit)
				}
				for _, elt := range n.Elts {
					if kv, ok := elt.(*ast.KeyValueExpr); ok {
						ast.Inspect(kv.Value, visit)
					} else {
						ast.Inspect(elt, visit)
					}
				}
				return visit(nil) // pop stack, don't descend
			}
		}
		return true
	}

	for _, f := range pkg.GetSyntax() {
		ast.Inspect(f, visit)
		if len(stack) != 0 {
			panic(stack)
		}
		if !ok {
			break
		}
	}
	return ok
}

// enclosingBlock returns the innermost block enclosing the specified
// AST node, specified in the form of a path from the root of the file,
// [file...n].
func enclosingBlock(info *types.Info, stack []ast.Node) *types.Scope {
	for i := range stack {
		n := stack[len(stack)-1-i]
		// For some reason, go/types always associates a
		// function's scope with its FuncType.
		// TODO(adonovan): feature or a bug?
		switch f := n.(type) {
		case *ast.FuncDecl:
			n = f.Type
		case *ast.FuncLit:
			n = f.Type
		}
		if b := info.Scopes[n]; b != nil {
			return b
		}
	}
	panic("no Scope for *ast.File")
}

func (r *renamer) checkLabel(label *types.Label) {
	// Check there are no identical labels in the function's label block.
	// (Label blocks don't nest, so this is easy.)
	if prev := label.Parent().Lookup(r.to); prev != nil {
		r.errorf(label.Pos(), "renaming this label %q to %q", label.Name(), prev.Name())
		r.errorf(prev.Pos(), "\twould conflict with this one")
	}
}

// checkStructField checks that the field renaming will not cause
// conflicts at its declaration, or ambiguity or changes to any selection.
func (r *renamer) checkStructField(from *types.Var) {

	// If this is the declaring package, check that the struct
	// declaration is free of field conflicts, and field/method
	// conflicts.
	//
	// go/types offers no easy way to get from a field (or interface
	// method) to its declaring struct (or interface), so we must
	// ascend the AST.
	if pgf, ok := enclosingFile(r.pkg, from.Pos()); ok {
		path, _ := astutil.PathEnclosingInterval(pgf.File, from.Pos(), from.Pos())
		// path matches this pattern:
		// [Ident SelectorExpr? StarExpr? Field FieldList StructType ParenExpr* ... File]

		// Ascend to FieldList.
		var i int
		for {
			if _, ok := path[i].(*ast.FieldList); ok {
				break
			}
			i++
		}
		i++
		tStruct := path[i].(*ast.StructType)
		i++
		// Ascend past parens (unlikely).
		for {
			_, ok := path[i].(*ast.ParenExpr)
			if !ok {
				break
			}
			i++
		}
		if spec, ok := path[i].(*ast.TypeSpec); ok {
			// This struct is also a named type.
			// We must check for direct (non-promoted) field/field
			// and method/field conflicts.
			named := r.pkg.GetTypesInfo().Defs[spec.Name].Type()
			prev, indices, _ := types.LookupFieldOrMethod(named, true, r.pkg.GetTypes(), r.to)
			if len(indices) == 1 {
				r.errorf(from.Pos(), "renaming this field %q to %q",
					from.Name(), r.to)
				r.errorf(prev.Pos(), "\twould conflict with this %s",
					objectKind(prev))
				return // skip checkSelections to avoid redundant errors
			}
		} else {
			// This struct is not a named type.
			// We need only check for direct (non-promoted) field/field conflicts.
			T := r.pkg.GetTypesInfo().Types[tStruct].Type.Underlying().(*types.Struct)
			for i := 0; i < T.NumFields(); i++ {
				if prev := T.Field(i); prev.Name() == r.to {
					r.errorf(from.Pos(), "renaming this field %q to %q",
						from.Name(), r.to)
					r.errorf(prev.Pos(), "\twould conflict with this field")
					return // skip checkSelections to avoid redundant errors
				}
			}
		}
	}

	// Renaming an anonymous field requires renaming the type too. e.g.
	// 	print(s.T)       // if we rename T to U,
	// 	type T int       // this and
	// 	var s struct {T} // this must change too.
	if from.Anonymous() {
		if named, ok := from.Type().(*types.Named); ok {
			r.check(named.Obj())
		} else if named, ok := Deref(from.Type()).(*types.Named); ok {
			r.check(named.Obj())
		}
	}

	// Check integrity of existing (field and method) selections.
	r.checkSelections(from)
}

// checkSelections checks that all uses and selections that resolve to
// the specified object would continue to do so after the renaming.
func (r *renamer) checkSelections(from types.Object) {
	pkg := r.pkg
	typ := pkg.GetTypes()
	{
		if id := someUse(pkg.GetTypesInfo(), from); id != nil {
			if !r.checkExport(id, typ, from) {
				return
			}
		}

		for syntax, sel := range pkg.GetTypesInfo().Selections {
			// There may be extant selections of only the old
			// name or only the new name, so we must check both.
			// (If neither, the renaming is sound.)
			//
			// In both cases, we wish to compare the lengths
			// of the implicit field path (Selection.Index)
			// to see if the renaming would change it.
			//
			// If a selection that resolves to 'from', when renamed,
			// would yield a path of the same or shorter length,
			// this indicates ambiguity or a changed referent,
			// analogous to same- or sub-block lexical conflict.
			//
			// If a selection using the name 'to' would
			// yield a path of the same or shorter length,
			// this indicates ambiguity or shadowing,
			// analogous to same- or super-block lexical conflict.

			// TODO(adonovan): fix: derive from Types[syntax.X].Mode
			// TODO(adonovan): test with pointer, value, addressable value.
			isAddressable := true

			if sel.Obj() == from {
				if obj, indices, _ := types.LookupFieldOrMethod(sel.Recv(), isAddressable, from.Pkg(), r.to); obj != nil {
					// Renaming this existing selection of
					// 'from' may block access to an existing
					// type member named 'to'.
					delta := len(indices) - len(sel.Index())
					if delta > 0 {
						continue // no ambiguity
					}
					r.selectionConflict(from, delta, syntax, obj)
					return
				}
			} else if sel.Obj().Name() == r.to {
				if obj, indices, _ := types.LookupFieldOrMethod(sel.Recv(), isAddressable, from.Pkg(), from.Name()); obj == from {
					// Renaming 'from' may cause this existing
					// selection of the name 'to' to change
					// its meaning.
					delta := len(indices) - len(sel.Index())
					if delta > 0 {
						continue //  no ambiguity
					}
					r.selectionConflict(from, -delta, syntax, sel.Obj())
					return
				}
			}
		}
	}
}

func (r *renamer) selectionConflict(from types.Object, delta int, syntax *ast.SelectorExpr, obj types.Object) {
	r.errorf(from.Pos(), "renaming this %s %q to %q",
		objectKind(from), from.Name(), r.to)

	switch {
	case delta < 0:
		// analogous to sub-block conflict
		r.errorf(syntax.Sel.Pos(),
			"\twould change the referent of this selection")
		r.errorf(obj.Pos(), "\tof this %s", objectKind(obj))
	case delta == 0:
		// analogous to same-block conflict
		r.errorf(syntax.Sel.Pos(),
			"\twould make this reference ambiguous")
		r.errorf(obj.Pos(), "\twith this %s", objectKind(obj))
	case delta > 0:
		// analogous to super-block conflict
		r.errorf(syntax.Sel.Pos(),
			"\twould shadow this selection")
		r.errorf(obj.Pos(), "\tof the %s declared here",
			objectKind(obj))
	}
}

// checkMethod performs safety checks for renaming a method.
// There are three hazards:
// - declaration conflicts
// - selection ambiguity/changes
// - entailed renamings of assignable concrete/interface types.
//
// We reject renamings initiated at concrete methods if it would
// change the assignability relation.  For renamings of abstract
// methods, we rename all methods transitively coupled to it via
// assignability.
func (r *renamer) checkMethod(from *types.Func) {
	// e.g. error.Error
	if from.Pkg() == nil {
		r.errorf(from.Pos(), "you cannot rename built-in method %s", from)
		return
	}

	// ASSIGNABILITY: We reject renamings of concrete methods that
	// would break a 'satisfy' constraint; but renamings of abstract
	// methods are allowed to proceed, and we rename affected
	// concrete and abstract methods as necessary.  It is the
	// initial method that determines the policy.

	// Check for conflict at point of declaration.
	// Check to ensure preservation of assignability requirements.
	R := recv(from).Type()
	if types.IsInterface(R) {
		// Abstract method

		// declaration
		prev, _, _ := types.LookupFieldOrMethod(R, false, from.Pkg(), r.to)
		if prev != nil {
			r.errorf(from.Pos(), "renaming this interface method %q to %q",
				from.Name(), r.to)
			r.errorf(prev.Pos(), "\twould conflict with this method")
			return
		}

		// Check all interfaces that embed this one for
		// declaration conflicts too.
		{
			// Start with named interface types (better errors)
			for _, obj := range r.pkg.GetTypesInfo().Defs {
				if obj, ok := obj.(*types.TypeName); ok && types.IsInterface(obj.Type()) {
					f, _, _ := types.LookupFieldOrMethod(
						obj.Type(), false, from.Pkg(), from.Name())
					if f == nil {
						continue
					}
					t, _, _ := types.LookupFieldOrMethod(
						obj.Type(), false, from.Pkg(), r.to)
					if t == nil {
						continue
					}
					r.errorf(from.Pos(), "renaming this interface method %q to %q",
						from.Name(), r.to)
					r.errorf(t.Pos(), "\twould conflict with this method")
					r.errorf(obj.Pos(), "\tin named interface type %q", obj.Name())
				}
			}

			// Now look at all literal interface types (includes named ones again).
			for e, tv := range r.pkg.GetTypesInfo().Types {
				if e, ok := e.(*ast.InterfaceType); ok {
					_ = e
					_ = tv.Type.(*types.Interface)
					// TODO(adonovan): implement same check as above.
				}
			}
		}

		// assignability
		//
		// Find the set of concrete or abstract methods directly
		// coupled to abstract method 'from' by some
		// satisfy.Constraint, and rename them too.
		for key := range r.satisfy() {
			// key = (lhs, rhs) where lhs is always an interface.

			lsel := r.msets.MethodSet(key.LHS).Lookup(from.Pkg(), from.Name())
			if lsel == nil {
				continue
			}
			rmethods := r.msets.MethodSet(key.RHS)
			rsel := rmethods.Lookup(from.Pkg(), from.Name())
			if rsel == nil {
				continue
			}

			// If both sides have a method of this name,
			// and one of them is m, the other must be coupled.
			var coupled *types.Func
			switch from {
			case lsel.Obj():
				coupled = rsel.Obj().(*types.Func)
			case rsel.Obj():
				coupled = lsel.Obj().(*types.Func)
			default:
				continue
			}

			// We must treat concrete-to-interface
			// constraints like an implicit selection C.f of
			// each interface method I.f, and check that the
			// renaming leaves the selection unchanged and
			// unambiguous.
			//
			// Fun fact: the implicit selection of C.f
			// 	type I interface{f()}
			// 	type C struct{I}
			// 	func (C) g()
			//      var _ I = C{} // here
			// yields abstract method I.f.  This can make error
			// messages less than obvious.
			//
			if !types.IsInterface(key.RHS) {
				// The logic below was derived from checkSelections.

				rtosel := rmethods.Lookup(from.Pkg(), r.to)
				if rtosel != nil {
					rto := rtosel.Obj().(*types.Func)
					delta := len(rsel.Index()) - len(rtosel.Index())
					if delta < 0 {
						continue // no ambiguity
					}

					// TODO(adonovan): record the constraint's position.
					keyPos := token.NoPos

					r.errorf(from.Pos(), "renaming this method %q to %q",
						from.Name(), r.to)
					if delta == 0 {
						// analogous to same-block conflict
						r.errorf(keyPos, "\twould make the %s method of %s invoked via interface %s ambiguous",
							r.to, key.RHS, key.LHS)
						r.errorf(rto.Pos(), "\twith (%s).%s",
							recv(rto).Type(), r.to)
					} else {
						// analogous to super-block conflict
						r.errorf(keyPos, "\twould change the %s method of %s invoked via interface %s",
							r.to, key.RHS, key.LHS)
						r.errorf(coupled.Pos(), "\tfrom (%s).%s",
							recv(coupled).Type(), r.to)
						r.errorf(rto.Pos(), "\tto (%s).%s",
							recv(rto).Type(), r.to)
					}
					return // one error is enough
				}
			}

			if !r.changeMethods {
				// This should be unreachable.
				r.errorf(from.Pos(), "internal error: during renaming of abstract method %s", from)
				r.errorf(coupled.Pos(), "\tchangedMethods=false, coupled method=%s", coupled)
				r.errorf(from.Pos(), "\tPlease file a bug report")
				return
			}

			// Rename the coupled method to preserve assignability.
			r.check(coupled)
		}
	} else {
		// Concrete method

		// declaration
		prev, indices, _ := types.LookupFieldOrMethod(R, true, from.Pkg(), r.to)
		if prev != nil && len(indices) == 1 {
			r.errorf(from.Pos(), "renaming this method %q to %q",
				from.Name(), r.to)
			r.errorf(prev.Pos(), "\twould conflict with this %s",
				objectKind(prev))
			return
		}

		// assignability
		//
		// Find the set of abstract methods coupled to concrete
		// method 'from' by some satisfy.Constraint, and rename
		// them too.
		//
		// Coupling may be indirect, e.g. I.f <-> C.f via type D.
		//
		// 	type I interface {f()}
		//	type C int
		//	type (C) f()
		//	type D struct{C}
		//	var _ I = D{}
		//
		for key := range r.satisfy() {
			// key = (lhs, rhs) where lhs is always an interface.
			if types.IsInterface(key.RHS) {
				continue
			}
			rsel := r.msets.MethodSet(key.RHS).Lookup(from.Pkg(), from.Name())
			if rsel == nil || rsel.Obj() != from {
				continue // rhs does not have the method
			}
			lsel := r.msets.MethodSet(key.LHS).Lookup(from.Pkg(), from.Name())
			if lsel == nil {
				continue
			}
			imeth := lsel.Obj().(*types.Func)

			// imeth is the abstract method (e.g. I.f)
			// and key.RHS is the concrete coupling type (e.g. D).
			if !r.changeMethods {
				r.errorf(from.Pos(), "renaming this method %q to %q",
					from.Name(), r.to)
				var pos token.Pos
				var iface string

				I := recv(imeth).Type()
				if named, ok := I.(*types.Named); ok {
					pos = named.Obj().Pos()
					iface = "interface " + named.Obj().Name()
				} else {
					pos = from.Pos()
					iface = I.String()
				}
				r.errorf(pos, "\twould make %s no longer assignable to %s",
					key.RHS, iface)
				r.errorf(imeth.Pos(), "\t(rename %s.%s if you intend to change both types)",
					I, from.Name())
				return // one error is enough
			}

			// Rename the coupled interface method to preserve assignability.
			r.check(imeth)
		}
	}

	// Check integrity of existing (field and method) selections.
	// We skip this if there were errors above, to avoid redundant errors.
	r.checkSelections(from)
}

func (r *renamer) checkExport(id *ast.Ident, pkg *types.Package, from types.Object) bool {
	// Reject cross-package references if r.to is unexported.
	// (Such references may be qualified identifiers or field/method
	// selections.)
	if !ast.IsExported(r.to) && pkg != from.Pkg() {
		r.errorf(from.Pos(),
			"renaming %q to %q would make it unexported",
			from.Name(), r.to)
		r.errorf(id.Pos(), "\tbreaking references from packages such as %q",
			pkg.Path())
		return false
	}
	return true
}

// satisfy returns the set of interface satisfaction constraints.
func (r *renamer) satisfy() map[satisfy.Constraint]bool {
	if r.satisfyConstraints == nil {
		// Compute on demand: it's expensive.
		var f satisfy.Finder
		pkg := r.pkg
		{
			// From satisfy.Finder documentation:
			//
			// The package must be free of type errors, and
			// info.{Defs,Uses,Selections,Types} must have been populated by the
			// type-checker.
			//
			// Only proceed if all packages have no errors.
			if pkg.HasParseErrors() || pkg.HasTypeErrors() {
				r.errorf(token.NoPos, // we don't have a position for this error.
					"renaming %q to %q not possible because %q has errors",
					r.from, r.to, pkg.Metadata().PkgPath)
				return nil
			}
			f.Find(pkg.GetTypesInfo(), pkg.GetSyntax())
		}
		r.satisfyConstraints = f.Result
	}
	return r.satisfyConstraints
}

// -- helpers ----------------------------------------------------------

// recv returns the method's receiver.
func recv(meth *types.Func) *types.Var {
	return meth.Type().(*types.Signature).Recv()
}

// someUse returns an arbitrary use of obj within info.
func someUse(info *types.Info, obj types.Object) *ast.Ident {
	for id, o := range info.Uses {
		if o == obj {
			return id
		}
	}
	return nil
}

func objectKind(obj types.Object) string {
	if obj == nil {
		return "nil object"
	}
	switch obj := obj.(type) {
	case *types.PkgName:
		return "imported package name"
	case *types.TypeName:
		return "type"
	case *types.Var:
		if obj.IsField() {
			return "field"
		}
	case *types.Func:
		if obj.Type().(*types.Signature).Recv() != nil {
			return "method"
		}
	}
	// label, func, var, const
	return strings.ToLower(strings.TrimPrefix(reflect.TypeOf(obj).String(), "*types."))
}

// NB: for renamings, blank is not considered valid.
func isValidIdentifier(id string) bool {
	if id == "" || id == "_" {
		return false
	}
	for i, r := range id {
		if !isLetter(r) && (i == 0 || !isDigit(r)) {
			return false
		}
	}
	return token.Lookup(id) == token.IDENT
}

// isLocal reports whether obj is local to some function.
// Precondition: not a struct field or interface method.
func isLocal(obj types.Object) bool {
	// [... 5=stmt 4=func 3=file 2=pkg 1=universe]
	var depth int
	for scope := obj.Parent(); scope != nil; scope = scope.Parent() {
		depth++
	}
	return depth >= 4
}

func isPackageLevel(obj types.Object) bool {
	if obj == nil {
		return false
	}
	return obj.Pkg().Scope().Lookup(obj.Name()) == obj
}

// -- Plundered from go/scanner: ---------------------------------------

func isLetter(ch rune) bool {
	return 'a' <= ch && ch <= 'z' || 'A' <= ch && ch <= 'Z' || ch == '_' || ch >= 0x80 && unicode.IsLetter(ch)
}

func isDigit(ch rune) bool {
	return '0' <= ch && ch <= '9' || ch >= 0x80 && unicode.IsDigit(ch)
}
