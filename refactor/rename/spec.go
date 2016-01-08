// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.5

package rename

// This file contains logic related to specifying a renaming: parsing of
// the flags as a form of query, and finding the object(s) it denotes.
// See Usage for flag details.

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"go/types"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"golang.org/x/tools/go/buildutil"
	"golang.org/x/tools/go/loader"
)

// A spec specifies an entity to rename.
//
// It is populated from an -offset flag or -from query;
// see Usage for the allowed -from query forms.
//
type spec struct {
	// pkg is the package containing the position
	// specified by the -from or -offset flag.
	// If filename == "", our search for the 'from' entity
	// is restricted to this package.
	pkg string

	// The original name of the entity being renamed.
	// If the query had a ::from component, this is that;
	// otherwise it's the last segment, e.g.
	//   (encoding/json.Decoder).from
	//   encoding/json.from
	fromName string

	// -- The remaining fields are private to this file.  All are optional. --

	// The query's ::x suffix, if any.
	searchFor string

	// e.g. "Decoder" in "(encoding/json.Decoder).fieldOrMethod"
	//                or "encoding/json.Decoder
	pkgMember string

	// e.g. fieldOrMethod in "(encoding/json.Decoder).fieldOrMethod"
	typeMember string

	// Restricts the query to this file.
	// Implied by -from="file.go::x" and -offset flags.
	filename string

	// Byte offset of the 'from' identifier within the file named 'filename'.
	// -offset mode only.
	offset int
}

// parseFromFlag interprets the "-from" flag value as a renaming specification.
// See Usage in rename.go for valid formats.
func parseFromFlag(ctxt *build.Context, fromFlag string) (*spec, error) {
	var spec spec
	var main string // sans "::x" suffix
	switch parts := strings.Split(fromFlag, "::"); len(parts) {
	case 1:
		main = parts[0]
	case 2:
		main = parts[0]
		spec.searchFor = parts[1]
		if parts[1] == "" {
			// error
		}
	default:
		return nil, fmt.Errorf("-from %q: invalid identifier specification (see -help for formats)", fromFlag)
	}

	if strings.HasSuffix(main, ".go") {
		// main is "filename.go"
		if spec.searchFor == "" {
			return nil, fmt.Errorf("-from: filename %q must have a ::name suffix", main)
		}
		spec.filename = main
		if !buildutil.FileExists(ctxt, spec.filename) {
			return nil, fmt.Errorf("no such file: %s", spec.filename)
		}

		bp, err := buildutil.ContainingPackage(ctxt, wd, spec.filename)
		if err != nil {
			return nil, err
		}
		spec.pkg = bp.ImportPath

	} else {
		// main is one of:
		//  "importpath"
		//  "importpath".member
		//  (*"importpath".type).fieldormethod           (parens and star optional)
		if err := parseObjectSpec(&spec, main); err != nil {
			return nil, err
		}
	}

	if spec.searchFor != "" {
		spec.fromName = spec.searchFor
	}

	cwd, err := os.Getwd()
	if err != nil {
		return nil, err
	}

	// Sanitize the package.
	bp, err := ctxt.Import(spec.pkg, cwd, build.FindOnly)
	if err != nil {
		return nil, fmt.Errorf("can't find package %q", spec.pkg)
	}
	spec.pkg = bp.ImportPath

	if !isValidIdentifier(spec.fromName) {
		return nil, fmt.Errorf("-from: invalid identifier %q", spec.fromName)
	}

	if Verbose {
		log.Printf("-from spec: %+v", spec)
	}

	return &spec, nil
}

// parseObjectSpec parses main as one of the non-filename forms of
// object specification.
func parseObjectSpec(spec *spec, main string) error {
	// Parse main as a Go expression, albeit a strange one.
	e, _ := parser.ParseExpr(main)

	if pkg := parseImportPath(e); pkg != "" {
		// e.g. bytes or "encoding/json": a package
		spec.pkg = pkg
		if spec.searchFor == "" {
			return fmt.Errorf("-from %q: package import path %q must have a ::name suffix",
				main, main)
		}
		return nil
	}

	if e, ok := e.(*ast.SelectorExpr); ok {
		x := unparen(e.X)

		// Strip off star constructor, if any.
		if star, ok := x.(*ast.StarExpr); ok {
			x = star.X
		}

		if pkg := parseImportPath(x); pkg != "" {
			// package member e.g. "encoding/json".HTMLEscape
			spec.pkg = pkg              // e.g. "encoding/json"
			spec.pkgMember = e.Sel.Name // e.g. "HTMLEscape"
			spec.fromName = e.Sel.Name
			return nil
		}

		if x, ok := x.(*ast.SelectorExpr); ok {
			// field/method of type e.g. ("encoding/json".Decoder).Decode
			y := unparen(x.X)
			if pkg := parseImportPath(y); pkg != "" {
				spec.pkg = pkg               // e.g. "encoding/json"
				spec.pkgMember = x.Sel.Name  // e.g. "Decoder"
				spec.typeMember = e.Sel.Name // e.g. "Decode"
				spec.fromName = e.Sel.Name
				return nil
			}
		}
	}

	return fmt.Errorf("-from %q: invalid expression", main)
}

// parseImportPath returns the import path of the package denoted by e.
// Any import path may be represented as a string literal;
// single-segment import paths (e.g. "bytes") may also be represented as
// ast.Ident.  parseImportPath returns "" for all other expressions.
func parseImportPath(e ast.Expr) string {
	switch e := e.(type) {
	case *ast.Ident:
		return e.Name // e.g. bytes

	case *ast.BasicLit:
		if e.Kind == token.STRING {
			pkgname, _ := strconv.Unquote(e.Value)
			return pkgname // e.g. "encoding/json"
		}
	}
	return ""
}

// parseOffsetFlag interprets the "-offset" flag value as a renaming specification.
func parseOffsetFlag(ctxt *build.Context, offsetFlag string) (*spec, error) {
	var spec spec
	// Validate -offset, e.g. file.go:#123
	parts := strings.Split(offsetFlag, ":#")
	if len(parts) != 2 {
		return nil, fmt.Errorf("-offset %q: invalid offset specification", offsetFlag)
	}

	spec.filename = parts[0]
	if !buildutil.FileExists(ctxt, spec.filename) {
		return nil, fmt.Errorf("no such file: %s", spec.filename)
	}

	bp, err := buildutil.ContainingPackage(ctxt, wd, spec.filename)
	if err != nil {
		return nil, err
	}
	spec.pkg = bp.ImportPath

	for _, r := range parts[1] {
		if !isDigit(r) {
			return nil, fmt.Errorf("-offset %q: non-numeric offset", offsetFlag)
		}
	}
	spec.offset, err = strconv.Atoi(parts[1])
	if err != nil {
		return nil, fmt.Errorf("-offset %q: non-numeric offset", offsetFlag)
	}

	// Parse the file and check there's an identifier at that offset.
	fset := token.NewFileSet()
	f, err := buildutil.ParseFile(fset, ctxt, nil, wd, spec.filename, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("-offset %q: cannot parse file: %s", offsetFlag, err)
	}

	id := identAtOffset(fset, f, spec.offset)
	if id == nil {
		return nil, fmt.Errorf("-offset %q: no identifier at this position", offsetFlag)
	}

	spec.fromName = id.Name

	return &spec, nil
}

var wd = func() string {
	wd, err := os.Getwd()
	if err != nil {
		panic("cannot get working directory: " + err.Error())
	}
	return wd
}()

// For source trees built with 'go build', the -from or -offset
// spec identifies exactly one initial 'from' object to rename ,
// but certain proprietary build systems allow a single file to
// appear in multiple packages (e.g. the test package contains a
// copy of its library), so there may be multiple objects for
// the same source entity.

func findFromObjects(iprog *loader.Program, spec *spec) ([]types.Object, error) {
	if spec.filename != "" {
		return findFromObjectsInFile(iprog, spec)
	}

	// Search for objects defined in specified package.

	// TODO(adonovan): the iprog.ImportMap has an entry {"main": ...}
	// for main packages, even though that's not an import path.
	// Seems like a bug.
	//
	// pkg := iprog.ImportMap[spec.pkg]
	// if pkg == nil {
	// 	return fmt.Errorf("cannot find package %s", spec.pkg) // can't happen?
	// }
	// info := iprog.AllPackages[pkg]

	// Workaround: lookup by value.
	var info *loader.PackageInfo
	var pkg *types.Package
	for pkg, info = range iprog.AllPackages {
		if pkg.Path() == spec.pkg {
			break
		}
	}
	if info == nil {
		return nil, fmt.Errorf("package %q was not loaded", spec.pkg)
	}

	objects, err := findObjects(info, spec)
	if err != nil {
		return nil, err
	}
	if len(objects) > 1 {
		// ambiguous "*" scope query
		return nil, ambiguityError(iprog.Fset, objects)
	}
	return objects, nil
}

func findFromObjectsInFile(iprog *loader.Program, spec *spec) ([]types.Object, error) {
	var fromObjects []types.Object
	for _, info := range iprog.AllPackages {
		// restrict to specified filename
		// NB: under certain proprietary build systems, a given
		// filename may appear in multiple packages.
		for _, f := range info.Files {
			thisFile := iprog.Fset.File(f.Pos())
			if !sameFile(thisFile.Name(), spec.filename) {
				continue
			}
			// This package contains the query file.

			if spec.offset != 0 {
				// Search for a specific ident by file/offset.
				id := identAtOffset(iprog.Fset, f, spec.offset)
				if id == nil {
					// can't happen?
					return nil, fmt.Errorf("identifier not found")
				}
				obj := info.Uses[id]
				if obj == nil {
					obj = info.Defs[id]
					if obj == nil {
						// Ident without Object.

						// Package clause?
						pos := thisFile.Pos(spec.offset)
						_, path, _ := iprog.PathEnclosingInterval(pos, pos)
						if len(path) == 2 { // [Ident File]
							// TODO(adonovan): support this case.
							return nil, fmt.Errorf("cannot rename %q: renaming package clauses is not yet supported",
								path[1].(*ast.File).Name.Name)
						}

						// Implicit y in "switch y := x.(type) {"?
						if obj := typeSwitchVar(&info.Info, path); obj != nil {
							return []types.Object{obj}, nil
						}

						// Probably a type error.
						return nil, fmt.Errorf("cannot find object for %q", id.Name)
					}
				}
				if obj.Pkg() == nil {
					return nil, fmt.Errorf("cannot rename predeclared identifiers (%s)", obj)

				}

				fromObjects = append(fromObjects, obj)
			} else {
				// do a package-wide query
				objects, err := findObjects(info, spec)
				if err != nil {
					return nil, err
				}

				// filter results: only objects defined in thisFile
				var filtered []types.Object
				for _, obj := range objects {
					if iprog.Fset.File(obj.Pos()) == thisFile {
						filtered = append(filtered, obj)
					}
				}
				if len(filtered) == 0 {
					return nil, fmt.Errorf("no object %q declared in file %s",
						spec.fromName, spec.filename)
				} else if len(filtered) > 1 {
					return nil, ambiguityError(iprog.Fset, filtered)
				}
				fromObjects = append(fromObjects, filtered[0])
			}
			break
		}
	}
	if len(fromObjects) == 0 {
		// can't happen?
		return nil, fmt.Errorf("file %s was not part of the loaded program", spec.filename)
	}
	return fromObjects, nil
}

func typeSwitchVar(info *types.Info, path []ast.Node) types.Object {
	if len(path) > 3 {
		// [Ident AssignStmt TypeSwitchStmt...]
		if sw, ok := path[2].(*ast.TypeSwitchStmt); ok {
			// choose the first case.
			if len(sw.Body.List) > 0 {
				obj := info.Implicits[sw.Body.List[0].(*ast.CaseClause)]
				if obj != nil {
					return obj
				}
			}
		}
	}
	return nil
}

// On success, findObjects returns the list of objects named
// spec.fromName matching the spec.  On success, the result has exactly
// one element unless spec.searchFor!="", in which case it has at least one
// element.
//
func findObjects(info *loader.PackageInfo, spec *spec) ([]types.Object, error) {
	if spec.pkgMember == "" {
		if spec.searchFor == "" {
			panic(spec)
		}
		objects := searchDefs(&info.Info, spec.searchFor)
		if objects == nil {
			return nil, fmt.Errorf("no object %q declared in package %q",
				spec.searchFor, info.Pkg.Path())
		}
		return objects, nil
	}

	pkgMember := info.Pkg.Scope().Lookup(spec.pkgMember)
	if pkgMember == nil {
		return nil, fmt.Errorf("package %q has no member %q",
			info.Pkg.Path(), spec.pkgMember)
	}

	var searchFunc *types.Func
	if spec.typeMember == "" {
		// package member
		if spec.searchFor == "" {
			return []types.Object{pkgMember}, nil
		}

		// Search within pkgMember, which must be a function.
		searchFunc, _ = pkgMember.(*types.Func)
		if searchFunc == nil {
			return nil, fmt.Errorf("cannot search for %q within %s %q",
				spec.searchFor, objectKind(pkgMember), pkgMember)
		}
	} else {
		// field/method of type
		// e.g. (encoding/json.Decoder).Decode
		// or ::x within it.

		tName, _ := pkgMember.(*types.TypeName)
		if tName == nil {
			return nil, fmt.Errorf("%s.%s is a %s, not a type",
				info.Pkg.Path(), pkgMember.Name(), objectKind(pkgMember))
		}

		// search within named type.
		obj, _, _ := types.LookupFieldOrMethod(tName.Type(), true, info.Pkg, spec.typeMember)
		if obj == nil {
			return nil, fmt.Errorf("cannot find field or method %q of %s %s.%s",
				spec.typeMember, typeKind(tName.Type()), info.Pkg.Path(), tName.Name())
		}

		if spec.searchFor == "" {
			// If it is an embedded field, return the type of the field.
			if v, ok := obj.(*types.Var); ok && v.Anonymous() {
				switch t := v.Type().(type) {
				case *types.Pointer:
					return []types.Object{t.Elem().(*types.Named).Obj()}, nil
				case *types.Named:
					return []types.Object{t.Obj()}, nil
				}
			}
			return []types.Object{obj}, nil
		}

		searchFunc, _ = obj.(*types.Func)
		if searchFunc == nil {
			return nil, fmt.Errorf("cannot search for local name %q within %s (%s.%s).%s; need a function",
				spec.searchFor, objectKind(obj), info.Pkg.Path(), tName.Name(),
				obj.Name())
		}
		if isInterface(tName.Type()) {
			return nil, fmt.Errorf("cannot search for local name %q within abstract method (%s.%s).%s",
				spec.searchFor, info.Pkg.Path(), tName.Name(), searchFunc.Name())
		}
	}

	// -- search within function or method --

	decl := funcDecl(info, searchFunc)
	if decl == nil {
		return nil, fmt.Errorf("cannot find syntax for %s", searchFunc) // can't happen?
	}

	var objects []types.Object
	for _, obj := range searchDefs(&info.Info, spec.searchFor) {
		// We use positions, not scopes, to determine whether
		// the obj is within searchFunc.  This is clumsy, but the
		// alternative, using the types.Scope tree, doesn't
		// account for non-lexical objects like fields and
		// interface methods.
		if decl.Pos() <= obj.Pos() && obj.Pos() < decl.End() && obj != searchFunc {
			objects = append(objects, obj)
		}
	}
	if objects == nil {
		return nil, fmt.Errorf("no local definition of %q within %s",
			spec.searchFor, searchFunc)
	}
	return objects, nil
}

func funcDecl(info *loader.PackageInfo, fn *types.Func) *ast.FuncDecl {
	for _, f := range info.Files {
		for _, d := range f.Decls {
			if d, ok := d.(*ast.FuncDecl); ok && info.Defs[d.Name] == fn {
				return d
			}
		}
	}
	return nil
}

func searchDefs(info *types.Info, name string) []types.Object {
	var objects []types.Object
	for id, obj := range info.Defs {
		if obj == nil {
			// e.g. blank ident.
			// TODO(adonovan): but also implicit y in
			//    switch y := x.(type)
			// Needs some thought.
			continue
		}
		if id.Name == name {
			objects = append(objects, obj)
		}
	}
	return objects
}

func identAtOffset(fset *token.FileSet, f *ast.File, offset int) *ast.Ident {
	var found *ast.Ident
	ast.Inspect(f, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok {
			idpos := fset.Position(id.Pos()).Offset
			if idpos <= offset && offset < idpos+len(id.Name) {
				found = id
			}
		}
		return found == nil // keep traversing only until found
	})
	return found
}

// ambiguityError returns an error describing an ambiguous "*" scope query.
func ambiguityError(fset *token.FileSet, objects []types.Object) error {
	var buf bytes.Buffer
	for i, obj := range objects {
		if i > 0 {
			buf.WriteString(", ")
		}
		posn := fset.Position(obj.Pos())
		fmt.Fprintf(&buf, "%s at %s:%d",
			objectKind(obj), filepath.Base(posn.Filename), posn.Column)
	}
	return fmt.Errorf("ambiguous specifier %s matches %s",
		objects[0].Name(), buf.String())
}
