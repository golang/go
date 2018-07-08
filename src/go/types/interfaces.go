// Copyright 2017 The Go Authors. All rights reserved.
// Use of this src code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
)

// This file implements the collection of an interface's methods
// without relying on partially computed types of methods or interfaces
// for interface types declared at the package level.
//
// Because interfaces must not embed themselves, directly or indirectly,
// the method set of a valid interface can always be computed independent
// of any cycles that might exist via method signatures (see also issue #18395).
//
// Except for blank method name and interface cycle errors, no errors
// are reported. Affected methods or embedded interfaces are silently
// dropped. Subsequent type-checking of the interface will check
// signatures and embedded interfaces and report errors at that time.
//
// Only infoFromTypeLit should be called directly from code outside this file
// to compute an ifaceInfo.

// ifaceInfo describes the method set for an interface.
// The zero value for an ifaceInfo is a ready-to-use ifaceInfo representing
// the empty interface.
type ifaceInfo struct {
	explicits int           // number of explicitly declared methods
	methods   []*methodInfo // all methods, starting with explicitly declared ones in source order
}

// emptyIfaceInfo represents the ifaceInfo for the empty interface.
var emptyIfaceInfo ifaceInfo

func (info *ifaceInfo) String() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "interface{")
	for i, m := range info.methods {
		if i > 0 {
			fmt.Fprint(&buf, " ")
		}
		fmt.Fprint(&buf, m)
	}
	fmt.Fprintf(&buf, "}")
	return buf.String()
}

// methodInfo represents an interface method.
// At least one of src or fun must be non-nil.
// (Methods declared in the current package have a non-nil scope
// and src, and eventually a non-nil fun field; imported and pre-
// declared methods have a nil scope and src, and only a non-nil
// fun field.)
type methodInfo struct {
	scope *Scope     // scope of interface method; or nil
	src   *ast.Field // syntax tree representation of interface method; or nil
	fun   *Func      // corresponding fully type-checked method type; or nil
}

func (info *methodInfo) String() string {
	if info.fun != nil {
		return info.fun.name
	}
	return info.src.Names[0].Name
}

func (info *methodInfo) Pos() token.Pos {
	if info.fun != nil {
		return info.fun.Pos()
	}
	return info.src.Pos()
}

func (info *methodInfo) id(pkg *Package) string {
	if info.fun != nil {
		return info.fun.Id()
	}
	return Id(pkg, info.src.Names[0].Name)
}

// A methodInfoSet maps method ids to methodInfos.
// It is used to determine duplicate declarations.
// (A methodInfo set is the equivalent of an objset
// but for methodInfos rather than Objects.)
type methodInfoSet map[string]*methodInfo

// insert attempts to insert an method m into the method set s.
// If s already contains an alternative method alt with
// the same name, insert leaves s unchanged and returns alt.
// Otherwise it inserts m and returns nil.
func (s *methodInfoSet) insert(pkg *Package, m *methodInfo) *methodInfo {
	id := m.id(pkg)
	if alt := (*s)[id]; alt != nil {
		return alt
	}
	if *s == nil {
		*s = make(methodInfoSet)
	}
	(*s)[id] = m
	return nil
}

// like Checker.declareInSet but for method infos.
func (check *Checker) declareInMethodSet(mset *methodInfoSet, pos token.Pos, m *methodInfo) bool {
	if alt := mset.insert(check.pkg, m); alt != nil {
		check.errorf(pos, "%s redeclared", m)
		check.reportAltMethod(alt)
		return false
	}
	return true
}

// like Checker.reportAltDecl but for method infos.
func (check *Checker) reportAltMethod(m *methodInfo) {
	if pos := m.Pos(); pos.IsValid() {
		// We use "other" rather than "previous" here because
		// the first declaration seen may not be textually
		// earlier in the source.
		check.errorf(pos, "\tother declaration of %s", m) // secondary error, \t indented
	}
}

// infoFromTypeLit computes the method set for the given interface iface
// declared in scope.
// If a corresponding type name exists (tname != nil), it is used for
// cycle detection and to cache the method set.
// The result is the method set, or nil if there is a cycle via embedded
// interfaces. A non-nil result doesn't mean that there were no errors,
// but they were either reported (e.g., blank methods), or will be found
// (again) when computing the interface's type.
// If tname is not nil it must be the last element in path.
func (check *Checker) infoFromTypeLit(scope *Scope, iface *ast.InterfaceType, tname *TypeName, path []*TypeName) (info *ifaceInfo) {
	assert(iface != nil)

	// lazy-allocate interfaces map
	if check.interfaces == nil {
		check.interfaces = make(map[*TypeName]*ifaceInfo)
	}

	if trace {
		check.trace(iface.Pos(), "-- collect methods for %v (path = %s, objPath = %s)", iface, pathString(path), check.pathString())
		check.indent++
		defer func() {
			check.indent--
			check.trace(iface.Pos(), "=> %s", info)
		}()
	}

	// If the interface is named, check if we computed info already.
	//
	// This is not simply an optimization; we may run into stack
	// overflow with recursive interface declarations. Example:
	//
	//      type T interface {
	//              m() interface { T }
	//      }
	//
	// (Since recursive definitions can only be expressed via names,
	// it is sufficient to track named interfaces here.)
	//
	// While at it, use the same mechanism to detect cycles. (We still
	// have the path-based cycle check because we want to report the
	// entire cycle if present.)
	if tname != nil {
		assert(path[len(path)-1] == tname) // tname must be last path element
		var found bool
		if info, found = check.interfaces[tname]; found {
			if info == nil {
				// We have a cycle and use check.cycle to report it.
				// We are guaranteed that check.cycle also finds the
				// cycle because when infoFromTypeLit is called, any
				// tname that's already in check.interfaces was also
				// added to the path. (But the converse is not true:
				// A non-nil tname is always the last element in path.)
				ok := check.cycle(tname, path, true)
				assert(ok)
			}
			return
		}
		check.interfaces[tname] = nil // computation started but not complete
	}

	if iface.Methods.List == nil {
		// fast track for empty interface
		info = &emptyIfaceInfo
	} else {
		// (syntactically) non-empty interface
		info = new(ifaceInfo)

		// collect explicitly declared methods and embedded interfaces
		var mset methodInfoSet
		var embeddeds []*ifaceInfo
		var positions []token.Pos // entries correspond to positions of embeddeds; used for error reporting
		for _, f := range iface.Methods.List {
			if len(f.Names) > 0 {
				// We have a method with name f.Names[0].
				// (The parser ensures that there's only one method
				// and we don't care if a constructed AST has more.)

				// spec: "As with all method sets, in an interface type,
				// each method must have a unique non-blank name."
				if name := f.Names[0]; name.Name == "_" {
					check.errorf(name.Pos(), "invalid method name _")
					continue // ignore
				}

				m := &methodInfo{scope: scope, src: f}
				if check.declareInMethodSet(&mset, f.Pos(), m) {
					info.methods = append(info.methods, m)
				}
			} else {
				// We have an embedded interface and f.Type is its
				// (possibly qualified) embedded type name. Collect
				// it if it's a valid interface.
				var e *ifaceInfo
				switch ename := f.Type.(type) {
				case *ast.Ident:
					e = check.infoFromTypeName(scope, ename, path)
				case *ast.SelectorExpr:
					e = check.infoFromQualifiedTypeName(scope, ename)
				default:
					// The parser makes sure we only see one of the above.
					// Constructed ASTs may contain other (invalid) nodes;
					// we simply ignore them. The full type-checking pass
					// will report those as errors later.
				}
				if e != nil {
					embeddeds = append(embeddeds, e)
					positions = append(positions, f.Type.Pos())
				}
			}
		}
		info.explicits = len(info.methods)

		// collect methods of embedded interfaces
		for i, e := range embeddeds {
			pos := positions[i] // position of type name of embedded interface
			for _, m := range e.methods {
				if check.declareInMethodSet(&mset, pos, m) {
					info.methods = append(info.methods, m)
				}
			}
		}
	}

	// mark check.interfaces as complete
	assert(info != nil)
	if tname != nil {
		check.interfaces[tname] = info
	}

	return
}

// infoFromTypeName computes the method set for the given type name
// which must denote a type whose underlying type is an interface.
// The same result qualifications apply as for infoFromTypeLit.
// infoFromTypeName should only be called from infoFromTypeLit.
func (check *Checker) infoFromTypeName(scope *Scope, name *ast.Ident, path []*TypeName) *ifaceInfo {
	// A single call of infoFromTypeName handles a sequence of (possibly
	// recursive) type declarations connected via unqualified type names.
	// Each type declaration leading to another typename causes a "tail call"
	// (goto) of this function. The general scenario looks like this:
	//
	//      ...
	//      type Pn T        // previous declarations leading to T, path = [..., Pn]
	//      type T interface { T0; ... }  // T0 leads to call of infoFromTypeName
	//
	//      // infoFromTypeName(name = T0, path = [..., Pn, T])
	//      type T0 T1       // path = [..., Pn, T, T0]
	//      type T1 T2  <-+  // path = [..., Pn, T, T0, T1]
	//      type T2 ...   |  // path = [..., Pn, T, T0, T1, T2]
	//      type Tn T1  --+  // path = [..., Pn, T, T0, T1, T2, Tn] and T1 is in path => cycle
	//
	// infoFromTypeName returns nil when such a cycle is detected. But in
	// contrast to cycles involving interfaces, we must not report the
	// error for "type name only" cycles because they will be found again
	// during type-checking of embedded interfaces. Reporting those cycles
	// here would lead to double reporting. Cycles involving embedding are
	// not reported again later because type-checking of interfaces relies
	// on the ifaceInfos computed here which are cycle-free by design.
	//
	// Remember the path length to detect "type name only" cycles.
	start := len(path)

typenameLoop:
	// name must be a type name denoting a type whose underlying type is an interface
	_, obj := scope.LookupParent(name.Name, check.pos)
	if obj == nil {
		return nil
	}
	tname, _ := obj.(*TypeName)
	if tname == nil {
		return nil
	}

	// We have a type name. It may be predeclared (error type),
	// imported (dot import), or declared by a type declaration.
	// It may not be an interface (e.g., predeclared type int).
	// Resolve it by analyzing each possible case.

	// Abort but don't report an error if we have a "type name only"
	// cycle (see big function comment).
	if check.cycle(tname, path[start:], false) {
		return nil
	}

	// Abort and report an error if we have a general cycle.
	if check.cycle(tname, path, true) {
		return nil
	}

	path = append(path, tname)

	// If tname is a package-level type declaration, it must be
	// in the objMap. Follow the RHS of that declaration if so.
	// The RHS may be a literal type (likely case), or another
	// (possibly parenthesized and/or qualified) type name.
	// (The declaration may be an alias declaration, but it
	// doesn't matter for the purpose of determining the under-
	// lying interface.)
	if decl := check.objMap[tname]; decl != nil {
		switch typ := unparen(decl.typ).(type) {
		case *ast.Ident:
			// type tname T
			name = typ
			goto typenameLoop
		case *ast.SelectorExpr:
			// type tname p.T
			return check.infoFromQualifiedTypeName(decl.file, typ)
		case *ast.InterfaceType:
			// type tname interface{...}
			return check.infoFromTypeLit(decl.file, typ, tname, path)
		}
		// type tname X // and X is not an interface type
		return nil
	}

	// If tname is not a package-level declaration, in a well-typed
	// program it should be a predeclared (error type), imported (dot
	// import), or function local declaration. Either way, it should
	// have been fully declared before use, except if there is a direct
	// cycle, and direct cycles will be caught above. Also, the denoted
	// type should be an interface (e.g., int is not an interface).
	if typ := tname.typ; typ != nil {
		// typ should be an interface
		if ityp, _ := typ.Underlying().(*Interface); ityp != nil {
			return infoFromType(ityp)
		}
	}

	// In all other cases we have some error.
	return nil
}

// infoFromQualifiedTypeName computes the method set for the given qualified type name, or nil.
func (check *Checker) infoFromQualifiedTypeName(scope *Scope, qname *ast.SelectorExpr) *ifaceInfo {
	// see also Checker.selector
	name, _ := qname.X.(*ast.Ident)
	if name == nil {
		return nil
	}
	_, obj1 := scope.LookupParent(name.Name, check.pos)
	if obj1 == nil {
		return nil
	}
	pname, _ := obj1.(*PkgName)
	if pname == nil {
		return nil
	}
	assert(pname.pkg == check.pkg)
	obj2 := pname.imported.scope.Lookup(qname.Sel.Name)
	if obj2 == nil || !obj2.Exported() {
		return nil
	}
	tname, _ := obj2.(*TypeName)
	if tname == nil {
		return nil
	}
	ityp, _ := tname.typ.Underlying().(*Interface)
	if ityp == nil {
		return nil
	}
	return infoFromType(ityp)
}

// infoFromType computes the method set for the given interface type.
// The result is never nil.
func infoFromType(typ *Interface) *ifaceInfo {
	assert(typ.allMethods != nil) // typ must be completely set up

	// fast track for empty interface
	n := len(typ.allMethods)
	if n == 0 {
		return &emptyIfaceInfo
	}

	info := new(ifaceInfo)
	info.explicits = len(typ.methods)
	info.methods = make([]*methodInfo, n)

	// If there are no embedded interfaces, simply collect the
	// explicitly declared methods (optimization of common case).
	if len(typ.methods) == n {
		for i, m := range typ.methods {
			info.methods[i] = &methodInfo{fun: m}
		}
		return info
	}

	// Interface types have a separate list for explicitly declared methods
	// which shares its methods with the list of all (explicitly declared or
	// embedded) methods. Collect all methods in a set so we can separate
	// the embedded methods from the explicitly declared ones.
	all := make(map[*Func]bool, n)
	for _, m := range typ.allMethods {
		all[m] = true
	}
	assert(len(all) == n) // methods must be unique

	// collect explicitly declared methods
	info.methods = make([]*methodInfo, n)
	for i, m := range typ.methods {
		info.methods[i] = &methodInfo{fun: m}
		delete(all, m)
	}

	// collect remaining (embedded) methods
	i := len(typ.methods)
	for m := range all {
		info.methods[i] = &methodInfo{fun: m}
		i++
	}
	assert(i == n)

	return info
}
