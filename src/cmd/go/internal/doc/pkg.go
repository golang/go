// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import (
	"bufio"
	"bytes"
	"fmt"
	"go/ast"
	"go/build"
	"go/doc"
	"go/format"
	"go/parser"
	"go/printer"
	"go/token"
	"io"
	"io/fs"
	"log"
	"path/filepath"
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	punchedCardWidth = 80
	indent           = "    "
)

type Package struct {
	writer      io.Writer    // Destination for output.
	name        string       // Package name, json for encoding/json.
	userPath    string       // String the user used to find this package.
	pkg         *ast.Package // Parsed package.
	file        *ast.File    // Merged from all files in the package
	doc         *doc.Package
	build       *build.Package
	typedValue  map[*doc.Value]bool // Consts and vars related to types.
	constructor map[*doc.Func]bool  // Constructors.
	fs          *token.FileSet      // Needed for printing.
	buf         pkgBuffer
}

func (pkg *Package) ToText(w io.Writer, text, prefix, codePrefix string) {
	d := pkg.doc.Parser().Parse(text)
	pr := pkg.doc.Printer()
	pr.TextPrefix = prefix
	pr.TextCodePrefix = codePrefix
	w.Write(pr.Text(d))
}

// pkgBuffer is a wrapper for bytes.Buffer that prints a package clause the
// first time Write is called.
type pkgBuffer struct {
	pkg     *Package
	printed bool // Prevent repeated package clauses.
	bytes.Buffer
}

func (pb *pkgBuffer) Write(p []byte) (int, error) {
	pb.packageClause()
	return pb.Buffer.Write(p)
}

func (pb *pkgBuffer) packageClause() {
	if !pb.printed {
		pb.printed = true
		// Only show package clause for commands if requested explicitly.
		if pb.pkg.pkg.Name != "main" || showCmd {
			pb.pkg.packageClause()
		}
	}
}

type PackageError string // type returned by pkg.Fatalf.

func (p PackageError) Error() string {
	return string(p)
}

// prettyPath returns a version of the package path that is suitable for an
// error message. It obeys the import comment if present. Also, since
// pkg.build.ImportPath is sometimes the unhelpful "" or ".", it looks for a
// directory name in GOROOT or GOPATH if that happens.
func (pkg *Package) prettyPath() string {
	path := pkg.build.ImportComment
	if path == "" {
		path = pkg.build.ImportPath
	}
	if path != "." && path != "" {
		return path
	}
	// Convert the source directory into a more useful path.
	// Also convert everything to slash-separated paths for uniform handling.
	path = filepath.Clean(filepath.ToSlash(pkg.build.Dir))
	// Can we find a decent prefix?
	if buildCtx.GOROOT != "" {
		goroot := filepath.Join(buildCtx.GOROOT, "src")
		if p, ok := trim(path, filepath.ToSlash(goroot)); ok {
			return p
		}
	}
	for _, gopath := range splitGopath() {
		if p, ok := trim(path, filepath.ToSlash(gopath)); ok {
			return p
		}
	}
	return path
}

// trim trims the directory prefix from the path, paying attention
// to the path separator. If they are the same string or the prefix
// is not present the original is returned. The boolean reports whether
// the prefix is present. That path and prefix have slashes for separators.
func trim(path, prefix string) (string, bool) {
	if !strings.HasPrefix(path, prefix) {
		return path, false
	}
	if path == prefix {
		return path, true
	}
	if path[len(prefix)] == '/' {
		return path[len(prefix)+1:], true
	}
	return path, false // Textual prefix but not a path prefix.
}

// pkg.Fatalf is like log.Fatalf, but panics so it can be recovered in the
// main do function, so it doesn't cause an exit. Allows testing to work
// without running a subprocess. The log prefix will be added when
// logged in main; it is not added here.
func (pkg *Package) Fatalf(format string, args ...any) {
	panic(PackageError(fmt.Sprintf(format, args...)))
}

// parsePackage turns the build package we found into a parsed package
// we can then use to generate documentation.
func parsePackage(writer io.Writer, pkg *build.Package, userPath string) *Package {
	// include tells parser.ParseDir which files to include.
	// That means the file must be in the build package's GoFiles or CgoFiles
	// list only (no tag-ignored files, tests, swig or other non-Go files).
	include := func(info fs.FileInfo) bool {
		for _, name := range pkg.GoFiles {
			if name == info.Name() {
				return true
			}
		}
		for _, name := range pkg.CgoFiles {
			if name == info.Name() {
				return true
			}
		}
		return false
	}
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, pkg.Dir, include, parser.ParseComments)
	if err != nil {
		log.Fatal(err)
	}
	// Make sure they are all in one package.
	if len(pkgs) == 0 {
		log.Fatalf("no source-code package in directory %s", pkg.Dir)
	}
	if len(pkgs) > 1 {
		log.Fatalf("multiple packages in directory %s", pkg.Dir)
	}
	astPkg := pkgs[pkg.Name]

	// TODO: go/doc does not include typed constants in the constants
	// list, which is what we want. For instance, time.Sunday is of type
	// time.Weekday, so it is defined in the type but not in the
	// Consts list for the package. This prevents
	//	go doc time.Sunday
	// from finding the symbol. Work around this for now, but we
	// should fix it in go/doc.
	// A similar story applies to factory functions.
	mode := doc.AllDecls
	if showSrc {
		mode |= doc.PreserveAST // See comment for Package.emit.
	}
	docPkg := doc.New(astPkg, pkg.ImportPath, mode)
	typedValue := make(map[*doc.Value]bool)
	constructor := make(map[*doc.Func]bool)
	for _, typ := range docPkg.Types {
		docPkg.Consts = append(docPkg.Consts, typ.Consts...)
		docPkg.Vars = append(docPkg.Vars, typ.Vars...)
		docPkg.Funcs = append(docPkg.Funcs, typ.Funcs...)
		if isExported(typ.Name) {
			for _, value := range typ.Consts {
				typedValue[value] = true
			}
			for _, value := range typ.Vars {
				typedValue[value] = true
			}
			for _, fun := range typ.Funcs {
				// We don't count it as a constructor bound to the type
				// if the type itself is not exported.
				constructor[fun] = true
			}
		}
	}

	p := &Package{
		writer:      writer,
		name:        pkg.Name,
		userPath:    userPath,
		pkg:         astPkg,
		file:        ast.MergePackageFiles(astPkg, 0),
		doc:         docPkg,
		typedValue:  typedValue,
		constructor: constructor,
		build:       pkg,
		fs:          fset,
	}
	p.buf.pkg = p
	return p
}

func (pkg *Package) Printf(format string, args ...any) {
	fmt.Fprintf(&pkg.buf, format, args...)
}

func (pkg *Package) flush() {
	_, err := pkg.writer.Write(pkg.buf.Bytes())
	if err != nil {
		log.Fatal(err)
	}
	pkg.buf.Reset() // Not needed, but it's a flush.
}

var newlineBytes = []byte("\n\n") // We never ask for more than 2.

// newlines guarantees there are n newlines at the end of the buffer.
func (pkg *Package) newlines(n int) {
	for !bytes.HasSuffix(pkg.buf.Bytes(), newlineBytes[:n]) {
		pkg.buf.WriteRune('\n')
	}
}

// emit prints the node. If showSrc is true, it ignores the provided comment,
// assuming the comment is in the node itself. Otherwise, the go/doc package
// clears the stuff we don't want to print anyway. It's a bit of a magic trick.
func (pkg *Package) emit(comment string, node ast.Node) {
	if node != nil {
		var arg any = node
		if showSrc {
			// Need an extra little dance to get internal comments to appear.
			arg = &printer.CommentedNode{
				Node:     node,
				Comments: pkg.file.Comments,
			}
		}
		err := format.Node(&pkg.buf, pkg.fs, arg)
		if err != nil {
			log.Fatal(err)
		}
		if comment != "" && !showSrc {
			pkg.newlines(1)
			pkg.ToText(&pkg.buf, comment, indent, indent+indent)
			pkg.newlines(2) // Blank line after comment to separate from next item.
		} else {
			pkg.newlines(1)
		}
	}
}

// oneLineNode returns a one-line summary of the given input node.
func (pkg *Package) oneLineNode(node ast.Node) string {
	const maxDepth = 10
	return pkg.oneLineNodeDepth(node, maxDepth)
}

// oneLineNodeDepth returns a one-line summary of the given input node.
// The depth specifies the maximum depth when traversing the AST.
func (pkg *Package) oneLineNodeDepth(node ast.Node, depth int) string {
	const dotDotDot = "..."
	if depth == 0 {
		return dotDotDot
	}
	depth--

	switch n := node.(type) {
	case nil:
		return ""

	case *ast.GenDecl:
		// Formats const and var declarations.
		trailer := ""
		if len(n.Specs) > 1 {
			trailer = " " + dotDotDot
		}

		// Find the first relevant spec.
		typ := ""
		for i, spec := range n.Specs {
			valueSpec := spec.(*ast.ValueSpec) // Must succeed; we can't mix types in one GenDecl.

			// The type name may carry over from a previous specification in the
			// case of constants and iota.
			if valueSpec.Type != nil {
				typ = fmt.Sprintf(" %s", pkg.oneLineNodeDepth(valueSpec.Type, depth))
			} else if len(valueSpec.Values) > 0 {
				typ = ""
			}

			if !isExported(valueSpec.Names[0].Name) {
				continue
			}
			val := ""
			if i < len(valueSpec.Values) && valueSpec.Values[i] != nil {
				val = fmt.Sprintf(" = %s", pkg.oneLineNodeDepth(valueSpec.Values[i], depth))
			}
			return fmt.Sprintf("%s %s%s%s%s", n.Tok, valueSpec.Names[0], typ, val, trailer)
		}
		return ""

	case *ast.FuncDecl:
		// Formats func declarations.
		name := n.Name.Name
		recv := pkg.oneLineNodeDepth(n.Recv, depth)
		if len(recv) > 0 {
			recv = "(" + recv + ") "
		}
		fnc := pkg.oneLineNodeDepth(n.Type, depth)
		fnc = strings.TrimPrefix(fnc, "func")
		return fmt.Sprintf("func %s%s%s", recv, name, fnc)

	case *ast.TypeSpec:
		sep := " "
		if n.Assign.IsValid() {
			sep = " = "
		}
		tparams := pkg.formatTypeParams(n.TypeParams, depth)
		return fmt.Sprintf("type %s%s%s%s", n.Name.Name, tparams, sep, pkg.oneLineNodeDepth(n.Type, depth))

	case *ast.FuncType:
		var params []string
		if n.Params != nil {
			for _, field := range n.Params.List {
				params = append(params, pkg.oneLineField(field, depth))
			}
		}
		needParens := false
		var results []string
		if n.Results != nil {
			needParens = needParens || len(n.Results.List) > 1
			for _, field := range n.Results.List {
				needParens = needParens || len(field.Names) > 0
				results = append(results, pkg.oneLineField(field, depth))
			}
		}

		tparam := pkg.formatTypeParams(n.TypeParams, depth)
		param := joinStrings(params)
		if len(results) == 0 {
			return fmt.Sprintf("func%s(%s)", tparam, param)
		}
		result := joinStrings(results)
		if !needParens {
			return fmt.Sprintf("func%s(%s) %s", tparam, param, result)
		}
		return fmt.Sprintf("func%s(%s) (%s)", tparam, param, result)

	case *ast.StructType:
		if n.Fields == nil || len(n.Fields.List) == 0 {
			return "struct{}"
		}
		return "struct{ ... }"

	case *ast.InterfaceType:
		if n.Methods == nil || len(n.Methods.List) == 0 {
			return "interface{}"
		}
		return "interface{ ... }"

	case *ast.FieldList:
		if n == nil || len(n.List) == 0 {
			return ""
		}
		if len(n.List) == 1 {
			return pkg.oneLineField(n.List[0], depth)
		}
		return dotDotDot

	case *ast.FuncLit:
		return pkg.oneLineNodeDepth(n.Type, depth) + " { ... }"

	case *ast.CompositeLit:
		typ := pkg.oneLineNodeDepth(n.Type, depth)
		if len(n.Elts) == 0 {
			return fmt.Sprintf("%s{}", typ)
		}
		return fmt.Sprintf("%s{ %s }", typ, dotDotDot)

	case *ast.ArrayType:
		length := pkg.oneLineNodeDepth(n.Len, depth)
		element := pkg.oneLineNodeDepth(n.Elt, depth)
		return fmt.Sprintf("[%s]%s", length, element)

	case *ast.MapType:
		key := pkg.oneLineNodeDepth(n.Key, depth)
		value := pkg.oneLineNodeDepth(n.Value, depth)
		return fmt.Sprintf("map[%s]%s", key, value)

	case *ast.CallExpr:
		fnc := pkg.oneLineNodeDepth(n.Fun, depth)
		var args []string
		for _, arg := range n.Args {
			args = append(args, pkg.oneLineNodeDepth(arg, depth))
		}
		return fmt.Sprintf("%s(%s)", fnc, joinStrings(args))

	case *ast.UnaryExpr:
		return fmt.Sprintf("%s%s", n.Op, pkg.oneLineNodeDepth(n.X, depth))

	case *ast.Ident:
		return n.Name

	default:
		// As a fallback, use default formatter for all unknown node types.
		buf := new(strings.Builder)
		format.Node(buf, pkg.fs, node)
		s := buf.String()
		if strings.Contains(s, "\n") {
			return dotDotDot
		}
		return s
	}
}

func (pkg *Package) formatTypeParams(list *ast.FieldList, depth int) string {
	if list.NumFields() == 0 {
		return ""
	}
	var tparams []string
	for _, field := range list.List {
		tparams = append(tparams, pkg.oneLineField(field, depth))
	}
	return "[" + joinStrings(tparams) + "]"
}

// oneLineField returns a one-line summary of the field.
func (pkg *Package) oneLineField(field *ast.Field, depth int) string {
	var names []string
	for _, name := range field.Names {
		names = append(names, name.Name)
	}
	if len(names) == 0 {
		return pkg.oneLineNodeDepth(field.Type, depth)
	}
	return joinStrings(names) + " " + pkg.oneLineNodeDepth(field.Type, depth)
}

// joinStrings formats the input as a comma-separated list,
// but truncates the list at some reasonable length if necessary.
func joinStrings(ss []string) string {
	var n int
	for i, s := range ss {
		n += len(s) + len(", ")
		if n > punchedCardWidth {
			ss = append(ss[:i:i], "...")
			break
		}
	}
	return strings.Join(ss, ", ")
}

// printHeader prints a header for the section named s, adding a blank line on each side.
func (pkg *Package) printHeader(s string) {
	pkg.Printf("\n%s\n\n", s)
}

// constsDoc prints all const documentation, if any, including a header.
// The one argument is the valueDoc registry.
func (pkg *Package) constsDoc(printed map[*ast.GenDecl]bool) {
	var header bool
	for _, value := range pkg.doc.Consts {
		// Constants and variables come in groups, and valueDoc prints
		// all the items in the group. We only need to find one exported symbol.
		for _, name := range value.Names {
			if isExported(name) && !pkg.typedValue[value] {
				if !header {
					pkg.printHeader("CONSTANTS")
					header = true
				}
				pkg.valueDoc(value, printed)
				break
			}
		}
	}
}

// varsDoc prints all var documentation, if any, including a header.
// Printed is the valueDoc registry.
func (pkg *Package) varsDoc(printed map[*ast.GenDecl]bool) {
	var header bool
	for _, value := range pkg.doc.Vars {
		// Constants and variables come in groups, and valueDoc prints
		// all the items in the group. We only need to find one exported symbol.
		for _, name := range value.Names {
			if isExported(name) && !pkg.typedValue[value] {
				if !header {
					pkg.printHeader("VARIABLES")
					header = true
				}
				pkg.valueDoc(value, printed)
				break
			}
		}
	}
}

// funcsDoc prints all func documentation, if any, including a header.
func (pkg *Package) funcsDoc() {
	var header bool
	for _, fun := range pkg.doc.Funcs {
		if isExported(fun.Name) && !pkg.constructor[fun] {
			if !header {
				pkg.printHeader("FUNCTIONS")
				header = true
			}
			pkg.emit(fun.Doc, fun.Decl)
		}
	}
}

// funcsDoc prints all type documentation, if any, including a header.
func (pkg *Package) typesDoc() {
	var header bool
	for _, typ := range pkg.doc.Types {
		if isExported(typ.Name) {
			if !header {
				pkg.printHeader("TYPES")
				header = true
			}
			pkg.typeDoc(typ)
		}
	}
}

// packageDoc prints the docs for the package.
func (pkg *Package) packageDoc() {
	pkg.Printf("") // Trigger the package clause; we know the package exists.
	if showAll || !short {
		pkg.ToText(&pkg.buf, pkg.doc.Doc, "", indent)
		pkg.newlines(1)
	}

	switch {
	case showAll:
		printed := make(map[*ast.GenDecl]bool) // valueDoc registry
		pkg.constsDoc(printed)
		pkg.varsDoc(printed)
		pkg.funcsDoc()
		pkg.typesDoc()

	case pkg.pkg.Name == "main" && !showCmd:
		// Show only package docs for commands.
		return

	default:
		if !short {
			pkg.newlines(2) // Guarantee blank line before the components.
		}
		pkg.valueSummary(pkg.doc.Consts, false)
		pkg.valueSummary(pkg.doc.Vars, false)
		pkg.funcSummary(pkg.doc.Funcs, false)
		pkg.typeSummary()
	}

	if !short {
		pkg.bugs()
	}
}

// packageClause prints the package clause.
func (pkg *Package) packageClause() {
	if short {
		return
	}
	importPath := pkg.build.ImportComment
	if importPath == "" {
		importPath = pkg.build.ImportPath
	}

	// If we're using modules, the import path derived from module code locations wins.
	// If we did a file system scan, we knew the import path when we found the directory.
	// But if we started with a directory name, we never knew the import path.
	// Either way, we don't know it now, and it's cheap to (re)compute it.
	if usingModules {
		for _, root := range codeRoots() {
			if pkg.build.Dir == root.dir {
				importPath = root.importPath
				break
			}
			if strings.HasPrefix(pkg.build.Dir, root.dir+string(filepath.Separator)) {
				suffix := filepath.ToSlash(pkg.build.Dir[len(root.dir)+1:])
				if root.importPath == "" {
					importPath = suffix
				} else {
					importPath = root.importPath + "/" + suffix
				}
				break
			}
		}
	}

	pkg.Printf("package %s // import %q\n\n", pkg.name, importPath)
	if !usingModules && importPath != pkg.build.ImportPath {
		pkg.Printf("WARNING: package source is installed in %q\n", pkg.build.ImportPath)
	}
}

// valueSummary prints a one-line summary for each set of values and constants.
// If all the types in a constant or variable declaration belong to the same
// type they can be printed by typeSummary, and so can be suppressed here.
func (pkg *Package) valueSummary(values []*doc.Value, showGrouped bool) {
	var isGrouped map[*doc.Value]bool
	if !showGrouped {
		isGrouped = make(map[*doc.Value]bool)
		for _, typ := range pkg.doc.Types {
			if !isExported(typ.Name) {
				continue
			}
			for _, c := range typ.Consts {
				isGrouped[c] = true
			}
			for _, v := range typ.Vars {
				isGrouped[v] = true
			}
		}
	}

	for _, value := range values {
		if !isGrouped[value] {
			if decl := pkg.oneLineNode(value.Decl); decl != "" {
				pkg.Printf("%s\n", decl)
			}
		}
	}
}

// funcSummary prints a one-line summary for each function. Constructors
// are printed by typeSummary, below, and so can be suppressed here.
func (pkg *Package) funcSummary(funcs []*doc.Func, showConstructors bool) {
	for _, fun := range funcs {
		// Exported functions only. The go/doc package does not include methods here.
		if isExported(fun.Name) {
			if showConstructors || !pkg.constructor[fun] {
				pkg.Printf("%s\n", pkg.oneLineNode(fun.Decl))
			}
		}
	}
}

// typeSummary prints a one-line summary for each type, followed by its constructors.
func (pkg *Package) typeSummary() {
	for _, typ := range pkg.doc.Types {
		for _, spec := range typ.Decl.Specs {
			typeSpec := spec.(*ast.TypeSpec) // Must succeed.
			if isExported(typeSpec.Name.Name) {
				pkg.Printf("%s\n", pkg.oneLineNode(typeSpec))
				// Now print the consts, vars, and constructors.
				for _, c := range typ.Consts {
					if decl := pkg.oneLineNode(c.Decl); decl != "" {
						pkg.Printf(indent+"%s\n", decl)
					}
				}
				for _, v := range typ.Vars {
					if decl := pkg.oneLineNode(v.Decl); decl != "" {
						pkg.Printf(indent+"%s\n", decl)
					}
				}
				for _, constructor := range typ.Funcs {
					if isExported(constructor.Name) {
						pkg.Printf(indent+"%s\n", pkg.oneLineNode(constructor.Decl))
					}
				}
			}
		}
	}
}

// bugs prints the BUGS information for the package.
// TODO: Provide access to TODOs and NOTEs as well (very noisy so off by default)?
func (pkg *Package) bugs() {
	if pkg.doc.Notes["BUG"] == nil {
		return
	}
	pkg.Printf("\n")
	for _, note := range pkg.doc.Notes["BUG"] {
		pkg.Printf("%s: %v\n", "BUG", note.Body)
	}
}

// findValues finds the doc.Values that describe the symbol.
func (pkg *Package) findValues(symbol string, docValues []*doc.Value) (values []*doc.Value) {
	for _, value := range docValues {
		for _, name := range value.Names {
			if match(symbol, name) {
				values = append(values, value)
			}
		}
	}
	return
}

// findFuncs finds the doc.Funcs that describes the symbol.
func (pkg *Package) findFuncs(symbol string) (funcs []*doc.Func) {
	for _, fun := range pkg.doc.Funcs {
		if match(symbol, fun.Name) {
			funcs = append(funcs, fun)
		}
	}
	return
}

// findTypes finds the doc.Types that describes the symbol.
// If symbol is empty, it finds all exported types.
func (pkg *Package) findTypes(symbol string) (types []*doc.Type) {
	for _, typ := range pkg.doc.Types {
		if symbol == "" && isExported(typ.Name) || match(symbol, typ.Name) {
			types = append(types, typ)
		}
	}
	return
}

// findTypeSpec returns the ast.TypeSpec within the declaration that defines the symbol.
// The name must match exactly.
func (pkg *Package) findTypeSpec(decl *ast.GenDecl, symbol string) *ast.TypeSpec {
	for _, spec := range decl.Specs {
		typeSpec := spec.(*ast.TypeSpec) // Must succeed.
		if symbol == typeSpec.Name.Name {
			return typeSpec
		}
	}
	return nil
}

// symbolDoc prints the docs for symbol. There may be multiple matches.
// If symbol matches a type, output includes its methods factories and associated constants.
// If there is no top-level symbol, symbolDoc looks for methods that match.
func (pkg *Package) symbolDoc(symbol string) bool {
	found := false
	// Functions.
	for _, fun := range pkg.findFuncs(symbol) {
		// Symbol is a function.
		decl := fun.Decl
		pkg.emit(fun.Doc, decl)
		found = true
	}
	// Constants and variables behave the same.
	values := pkg.findValues(symbol, pkg.doc.Consts)
	values = append(values, pkg.findValues(symbol, pkg.doc.Vars)...)
	printed := make(map[*ast.GenDecl]bool) // valueDoc registry
	for _, value := range values {
		pkg.valueDoc(value, printed)
		found = true
	}
	// Types.
	for _, typ := range pkg.findTypes(symbol) {
		pkg.typeDoc(typ)
		found = true
	}
	if !found {
		// See if there are methods.
		if !pkg.printMethodDoc("", symbol) {
			return false
		}
	}
	return true
}

// valueDoc prints the docs for a constant or variable. The printed map records
// which values have been printed already to avoid duplication. Otherwise, a
// declaration like:
//
//	const ( c = 1; C = 2 )
//
// … could be printed twice if the -u flag is set, as it matches twice.
func (pkg *Package) valueDoc(value *doc.Value, printed map[*ast.GenDecl]bool) {
	if printed[value.Decl] {
		return
	}
	// Print each spec only if there is at least one exported symbol in it.
	// (See issue 11008.)
	// TODO: Should we elide unexported symbols from a single spec?
	// It's an unlikely scenario, probably not worth the trouble.
	// TODO: Would be nice if go/doc did this for us.
	specs := make([]ast.Spec, 0, len(value.Decl.Specs))
	var typ ast.Expr
	for _, spec := range value.Decl.Specs {
		vspec := spec.(*ast.ValueSpec)

		// The type name may carry over from a previous specification in the
		// case of constants and iota.
		if vspec.Type != nil {
			typ = vspec.Type
		}

		for _, ident := range vspec.Names {
			if showSrc || isExported(ident.Name) {
				if vspec.Type == nil && vspec.Values == nil && typ != nil {
					// This a standalone identifier, as in the case of iota usage.
					// Thus, assume the type comes from the previous type.
					vspec.Type = &ast.Ident{
						Name:    pkg.oneLineNode(typ),
						NamePos: vspec.End() - 1,
					}
				}

				specs = append(specs, vspec)
				typ = nil // Only inject type on first exported identifier
				break
			}
		}
	}
	if len(specs) == 0 {
		return
	}
	value.Decl.Specs = specs
	pkg.emit(value.Doc, value.Decl)
	printed[value.Decl] = true
}

// typeDoc prints the docs for a type, including constructors and other items
// related to it.
func (pkg *Package) typeDoc(typ *doc.Type) {
	decl := typ.Decl
	spec := pkg.findTypeSpec(decl, typ.Name)
	trimUnexportedElems(spec)
	// If there are multiple types defined, reduce to just this one.
	if len(decl.Specs) > 1 {
		decl.Specs = []ast.Spec{spec}
	}
	pkg.emit(typ.Doc, decl)
	pkg.newlines(2)
	// Show associated methods, constants, etc.
	if showAll {
		printed := make(map[*ast.GenDecl]bool) // valueDoc registry
		// We can use append here to print consts, then vars. Ditto for funcs and methods.
		values := typ.Consts
		values = append(values, typ.Vars...)
		for _, value := range values {
			for _, name := range value.Names {
				if isExported(name) {
					pkg.valueDoc(value, printed)
					break
				}
			}
		}
		funcs := typ.Funcs
		funcs = append(funcs, typ.Methods...)
		for _, fun := range funcs {
			if isExported(fun.Name) {
				pkg.emit(fun.Doc, fun.Decl)
				if fun.Doc == "" {
					pkg.newlines(2)
				}
			}
		}
	} else {
		pkg.valueSummary(typ.Consts, true)
		pkg.valueSummary(typ.Vars, true)
		pkg.funcSummary(typ.Funcs, true)
		pkg.funcSummary(typ.Methods, true)
	}
}

// trimUnexportedElems modifies spec in place to elide unexported fields from
// structs and methods from interfaces (unless the unexported flag is set or we
// are asked to show the original source).
func trimUnexportedElems(spec *ast.TypeSpec) {
	if showSrc {
		return
	}
	switch typ := spec.Type.(type) {
	case *ast.StructType:
		typ.Fields = trimUnexportedFields(typ.Fields, false)
	case *ast.InterfaceType:
		typ.Methods = trimUnexportedFields(typ.Methods, true)
	}
}

// trimUnexportedFields returns the field list trimmed of unexported fields.
func trimUnexportedFields(fields *ast.FieldList, isInterface bool) *ast.FieldList {
	what := "methods"
	if !isInterface {
		what = "fields"
	}

	trimmed := false
	list := make([]*ast.Field, 0, len(fields.List))
	for _, field := range fields.List {
		// When printing fields we normally print field.Doc.
		// Here we are going to pass the AST to go/format,
		// which will print the comments from the AST,
		// not field.Doc which is from go/doc.
		// The two are similar but not identical;
		// for example, field.Doc does not include directives.
		// In order to consistently print field.Doc,
		// we replace the comment in the AST with field.Doc.
		// That will cause go/format to print what we want.
		// See issue #56592.
		if field.Doc != nil {
			doc := field.Doc
			text := doc.Text()

			trailingBlankLine := len(doc.List[len(doc.List)-1].Text) == 2
			if !trailingBlankLine {
				// Remove trailing newline.
				lt := len(text)
				if lt > 0 && text[lt-1] == '\n' {
					text = text[:lt-1]
				}
			}

			start := doc.List[0].Slash
			doc.List = doc.List[:0]
			for line := range strings.SplitSeq(text, "\n") {
				prefix := "// "
				if len(line) > 0 && line[0] == '\t' {
					prefix = "//"
				}
				doc.List = append(doc.List, &ast.Comment{
					Text: prefix + line,
				})
			}
			doc.List[0].Slash = start
		}

		names := field.Names
		if len(names) == 0 {
			// Embedded type. Use the name of the type. It must be of the form ident or
			// pkg.ident (for structs and interfaces), or *ident or *pkg.ident (structs only).
			// Or a type embedded in a constraint.
			// Nothing else is allowed.
			ty := field.Type
			if se, ok := field.Type.(*ast.StarExpr); !isInterface && ok {
				// The form *ident or *pkg.ident is only valid on
				// embedded types in structs.
				ty = se.X
			}
			constraint := false
			switch ident := ty.(type) {
			case *ast.Ident:
				if isInterface && ident.Name == "error" && ident.Obj == nil {
					// For documentation purposes, we consider the builtin error
					// type special when embedded in an interface, such that it
					// always gets shown publicly.
					list = append(list, field)
					continue
				}
				names = []*ast.Ident{ident}
			case *ast.SelectorExpr:
				// An embedded type may refer to a type in another package.
				names = []*ast.Ident{ident.Sel}
			default:
				// An approximation or union or type
				// literal in an interface.
				constraint = true
			}
			if names == nil && !constraint {
				// Can only happen if AST is incorrect. Safe to continue with a nil list.
				log.Print("invalid program: unexpected type for embedded field")
			}
		}
		// Trims if any is unexported. Good enough in practice.
		ok := true
		if !unexported {
			for _, name := range names {
				if !isExported(name.Name) {
					trimmed = true
					ok = false
					break
				}
			}
		}
		if ok {
			list = append(list, field)
		}
	}
	if !trimmed {
		return fields
	}
	unexportedField := &ast.Field{
		Type: &ast.Ident{
			// Hack: printer will treat this as a field with a named type.
			// Setting Name and NamePos to ("", fields.Closing-1) ensures that
			// when Pos and End are called on this field, they return the
			// position right before closing '}' character.
			Name:    "",
			NamePos: fields.Closing - 1,
		},
		Comment: &ast.CommentGroup{
			List: []*ast.Comment{{Text: fmt.Sprintf("// Has unexported %s.\n", what)}},
		},
	}
	return &ast.FieldList{
		Opening: fields.Opening,
		List:    append(list, unexportedField),
		Closing: fields.Closing,
	}
}

// printMethodDoc prints the docs for matches of symbol.method.
// If symbol is empty, it prints all methods for any concrete type
// that match the name. It reports whether it found any methods.
func (pkg *Package) printMethodDoc(symbol, method string) bool {
	types := pkg.findTypes(symbol)
	if types == nil {
		if symbol == "" {
			return false
		}
		pkg.Fatalf("symbol %s is not a type in package %s installed in %q", symbol, pkg.name, pkg.build.ImportPath)
	}
	found := false
	for _, typ := range types {
		if len(typ.Methods) > 0 {
			for _, meth := range typ.Methods {
				if match(method, meth.Name) {
					decl := meth.Decl
					pkg.emit(meth.Doc, decl)
					found = true
				}
			}
			continue
		}
		if symbol == "" {
			continue
		}
		// Type may be an interface. The go/doc package does not attach
		// an interface's methods to the doc.Type. We need to dig around.
		spec := pkg.findTypeSpec(typ.Decl, typ.Name)
		inter, ok := spec.Type.(*ast.InterfaceType)
		if !ok {
			// Not an interface type.
			continue
		}

		// Collect and print only the methods that match.
		var methods []*ast.Field
		for _, iMethod := range inter.Methods.List {
			// This is an interface, so there can be only one name.
			// TODO: Anonymous methods (embedding)
			if len(iMethod.Names) == 0 {
				continue
			}
			name := iMethod.Names[0].Name
			if match(method, name) {
				methods = append(methods, iMethod)
				found = true
			}
		}
		if found {
			pkg.Printf("type %s ", spec.Name)
			inter.Methods.List, methods = methods, inter.Methods.List
			err := format.Node(&pkg.buf, pkg.fs, inter)
			if err != nil {
				log.Fatal(err)
			}
			pkg.newlines(1)
			// Restore the original methods.
			inter.Methods.List = methods
		}
	}
	return found
}

// printFieldDoc prints the docs for matches of symbol.fieldName.
// It reports whether it found any field.
// Both symbol and fieldName must be non-empty or it returns false.
func (pkg *Package) printFieldDoc(symbol, fieldName string) bool {
	if symbol == "" || fieldName == "" {
		return false
	}
	types := pkg.findTypes(symbol)
	if types == nil {
		pkg.Fatalf("symbol %s is not a type in package %s installed in %q", symbol, pkg.name, pkg.build.ImportPath)
	}
	found := false
	numUnmatched := 0
	for _, typ := range types {
		// Type must be a struct.
		spec := pkg.findTypeSpec(typ.Decl, typ.Name)
		structType, ok := spec.Type.(*ast.StructType)
		if !ok {
			// Not a struct type.
			continue
		}
		for _, field := range structType.Fields.List {
			// TODO: Anonymous fields.
			for _, name := range field.Names {
				if !match(fieldName, name.Name) {
					numUnmatched++
					continue
				}
				if !found {
					pkg.Printf("type %s struct {\n", typ.Name)
				}
				if field.Doc != nil {
					// To present indented blocks in comments correctly, process the comment as
					// a unit before adding the leading // to each line.
					docBuf := new(bytes.Buffer)
					pkg.ToText(docBuf, field.Doc.Text(), "", indent)
					scanner := bufio.NewScanner(docBuf)
					for scanner.Scan() {
						fmt.Fprintf(&pkg.buf, "%s// %s\n", indent, scanner.Bytes())
					}
				}
				s := pkg.oneLineNode(field.Type)
				lineComment := ""
				if field.Comment != nil {
					lineComment = fmt.Sprintf("  %s", field.Comment.List[0].Text)
				}
				pkg.Printf("%s%s %s%s\n", indent, name, s, lineComment)
				found = true
			}
		}
	}
	if found {
		if numUnmatched > 0 {
			pkg.Printf("\n    // ... other fields elided ...\n")
		}
		pkg.Printf("}\n")
	}
	return found
}

// match reports whether the user's symbol matches the program's.
// A lower-case character in the user's string matches either case in the program's.
// The program string must be exported.
func match(user, program string) bool {
	if !isExported(program) {
		return false
	}
	if matchCase {
		return user == program
	}
	for _, u := range user {
		p, w := utf8.DecodeRuneInString(program)
		program = program[w:]
		if u == p {
			continue
		}
		if unicode.IsLower(u) && simpleFold(u) == simpleFold(p) {
			continue
		}
		return false
	}
	return program == ""
}

// simpleFold returns the minimum rune equivalent to r
// under Unicode-defined simple case folding.
func simpleFold(r rune) rune {
	for {
		r1 := unicode.SimpleFold(r)
		if r1 <= r {
			return r1 // wrapped around, found min
		}
		r = r1
	}
}
