// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"io"
	"maps"
	"slices"
	"strconv"
	"strings"
)

// genBindingFile is an input file for -gen-binding.
type genBindingFile struct {
	src     []byte
	srcDir  string
	ast     *ast.File // parsed with comments
	pragmas []cgoPragma
	// importsC reports whether the file imports "C". The generated binding
	// file replaces the files that do; the others stay in the package.
	importsC bool
	// userDecls maps the Go names declared in the file to the source text
	// of their declarations (with their doc comments), or to "" if a
	// declaration cannot be kept on its own. Methods are keyed by "T.Name".
	userDecls map[string]string
	userPos   map[string]token.Pos
	valueRecv map[string]bool // methods declared with a non-pointer receiver
}

// isAnonCtype reports whether name is the _Ctype_ name cgo invents for an
// untagged C struct, union, or class, such as _Ctype_struct___0. Such names
// do not name C types, so they are never bound by //cgo:type pragmas.
func isAnonCtype(name string) bool {
	for _, prefix := range []string{"_Ctype_struct___", "_Ctype_union___", "_Ctype_class___"} {
		if n, ok := strings.CutPrefix(name, prefix); ok && n != "" && strings.Trim(n, "0123456789") == "" {
			return true
		}
	}
	return false
}

func exprUsesIdent(expr ast.Expr, name string) bool {
	found := false
	ast.Inspect(expr, func(n ast.Node) bool {
		if found {
			return false
		}
		if id, ok := n.(*ast.Ident); ok && id.Name == name {
			found = true
			return false
		}
		return true
	})
	return found
}

// genBindingImportsC reports whether the Go source src imports "C". The
// input files of -gen-binding that do not are read only for the Go
// declarations they provide.
func genBindingImportsC(input string, src []byte) bool {
	file, err := parser.ParseFile(token.NewFileSet(), input, src, parser.ImportsOnly)
	return err != nil || hasImportCFile(file) // let ParseGo report errors
}

// useGenBindingPragmas prepares f for -gen-binding: the //cgo: pragmas in
// the file, not its C.name references, say which C declarations to bind.
// It replaces the names collected by ParseGo with the C names the pragmas
// refer to, so that they (and the C types they depend on) are looked up in
// the C headers.
func (f *File) useGenBindingPragmas(input, origDir string, src []byte) {
	file, err := parser.ParseFile(fset, input, src, parser.SkipObjectResolution|parser.ParseComments)
	if err != nil {
		fatalf("%s", err)
	}
	g := &genBindingFile{
		src:       src,
		srcDir:    origDir,
		ast:       file,
		pragmas:   parseCgoPragmas(fset, file),
		importsC:  hasImportCFile(file),
		userDecls: make(map[string]string),
		userPos:   make(map[string]token.Pos),
		valueRecv: make(map[string]bool),
	}
	f.gen = g

	text := func(n ast.Node) string {
		return string(src[fset.Position(n.Pos()).Offset:fset.Position(n.End()).Offset])
	}
	// withDoc returns decl preceded by its doc comment, without any
	// //cgo: pragmas in it: the output lists those separately.
	withDoc := func(doc *ast.CommentGroup, decl string) string {
		if doc == nil {
			return decl
		}
		var b strings.Builder
		for _, c := range doc.List {
			if !isCgoPragmaComment(c.Text) {
				b.WriteString(c.Text + "\n")
			}
		}
		return b.String() + decl
	}
	declare := func(name, decl string, pos token.Pos) {
		g.userDecls[name] = decl
		g.userPos[name] = pos
	}
	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			name := d.Name.Name
			if d.Recv != nil {
				// A method, possibly of a //cgo:dynload type.
				if len(d.Recv.List) != 1 {
					continue
				}
				recv := d.Recv.List[0].Type
				star, isPtr := recv.(*ast.StarExpr)
				if isPtr {
					recv = star.X
				}
				id, ok := recv.(*ast.Ident)
				if !ok {
					continue
				}
				name = id.Name + "." + name
				g.valueRecv[name] = !isPtr
			}
			if d.Body != nil {
				declare(name, "", d.Pos())
			} else {
				declare(name, withDoc(d.Doc, text(d)), d.Pos())
			}
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				var doc *ast.CommentGroup
				if !d.Lparen.IsValid() {
					doc = d.Doc
				}
				switch s := spec.(type) {
				case *ast.TypeSpec:
					if s.Doc != nil {
						doc = s.Doc
					}
					declare(s.Name.Name, withDoc(doc, "type "+text(s)), s.Pos())
				case *ast.ValueSpec:
					if s.Doc != nil {
						doc = s.Doc
					}
					invalidConst := d.Tok == token.CONST && (len(s.Values) == 0 || slices.ContainsFunc(s.Values, func(e ast.Expr) bool { return exprUsesIdent(e, "iota") }))
					for _, id := range s.Names {
						if len(s.Names) != 1 || invalidConst {
							declare(id.Name, "", s.Pos())
						} else {
							declare(id.Name, withDoc(doc, d.Tok.String()+" "+text(s)), s.Pos())
						}
					}
				}
			}
		}
	}

	f.Ref = nil
	f.Calls = nil
	f.Name = make(map[string]*Name)
	f.NamePos = make(map[*Name]token.Pos)
	for _, pr := range g.pragmas {
		if pr.Kind == "dynload" {
			continue
		}
		if f.Name[pr.CName] == nil {
			n := &Name{Go: pr.CName, C: cname(pr.CName)}
			f.Name[pr.CName] = n
			f.NamePos[n] = pr.TokPos
		}
	}
}

// writeBinding writes the Go binding file generated for the -gen-binding
// input files to w: their C preambles and pragmas, the Go declarations the
// pragmas name, and //cgo:type pragmas and declarations for the C types
// those declarations depend on.
func (p *Package) writeBinding(w io.Writer) {
	// Merge the pragmas and declarations of the input files.
	pkgName := p.genFiles[0].gen.ast.Name
	// The output replaces the files that import "C", so it keeps the
	// declarations in them. The other files stay in the package next to
	// the output: their declarations are left alone (not generated or
	// copied), and the names invented for C types avoid them.
	var allPragmas []cgoPragma
	userDecls := make(map[string]string)
	userPos := make(map[string]token.Pos)
	valueRecv := make(map[string]bool)
	declaredTwice := make(map[string]bool)
	otherDecls := make(map[string]bool) // declared in files without import "C"
	for _, f := range p.genFiles {
		g := f.gen
		if g.ast.Name.Name != pkgName.Name {
			error_(g.ast.Name.Pos(), "-gen-binding: package %s; expected package %s", g.ast.Name.Name, pkgName.Name)
		}
		allPragmas = append(allPragmas, g.pragmas...)
		for name, decl := range g.userDecls {
			if _, ok := userDecls[name]; ok {
				declaredTwice[name] = true
			}
			userDecls[name] = decl
			userPos[name] = g.userPos[name]
			valueRecv[name] = g.valueRecv[name]
			otherDecls[name] = !g.importsC
		}
	}
	// declKey returns the key of pr's Go declaration in userDecls.
	declKey := func(pr cgoPragma) string {
		if pr.RecvType != "" {
			return pr.RecvType + "." + pr.GoName
		}
		return pr.GoName
	}

	// Check that no Go or C. name is bound twice, that method pragmas
	// name //cgo:dynload types, and that the declarations the input
	// provides can be kept as written.
	dynloadTypes := make(map[string]bool)
	for _, pr := range allPragmas {
		if pr.Kind == "dynload" {
			dynloadTypes[pr.GoName] = true
		}
	}
	boundGo := make(map[string]bool)
	boundC := make(map[string]bool)
	for _, pr := range allPragmas {
		key := declKey(pr)
		if pr.RecvType != "" && !dynloadTypes[pr.RecvType] {
			error_(pr.TokPos, "//cgo:%s %s: %s is not declared with //cgo:dynload", pr.Kind, key, pr.RecvType)
		}
		if pr.BindC {
			if boundC[pr.CName] {
				error_(pr.TokPos, "C.%s is bound by more than one //cgo: pragma", pr.CName)
			}
			boundC[pr.CName] = true
		}
		if boundGo[key] {
			error_(pr.TokPos, "%s is bound by more than one //cgo: pragma", key)
			continue
		}
		boundGo[key] = true
		if declaredTwice[key] {
			error_(userPos[key], "-gen-binding: %s is declared more than once", key)
		} else if decl, ok := userDecls[key]; ok && decl == "" {
			error_(userPos[key], "-gen-binding: %s must be declared on its own (without a body, iota, or omitted const value), or not at all", key)
		} else if ok && valueRecv[key] {
			error_(userPos[key], "-gen-binding: //cgo:%s method %s must have a pointer receiver (*%s)", pr.Kind, pr.GoName, pr.RecvType)
		}
	}
	if nerrors > 0 {
		return
	}

	// Check that each pragma names a C declaration of the right kind.
	for _, pr := range allPragmas {
		if pr.Kind == "dynload" {
			continue
		}
		n := p.Name[pr.CName]
		var ok bool
		var want string
		switch pr.Kind {
		case "func":
			ok, want = n != nil && n.FuncType != nil, "function"
		case "var":
			ok, want = n != nil && n.IsVar(), "variable"
		case "type":
			ok, want = n != nil && n.Kind == "type", "type"
		case "const":
			ok, want = n != nil && n.IsConst() && n.Const != "", "constant"
		}
		if !ok {
			kind := "not declared"
			if n != nil && n.Kind != "" {
				kind = "a " + n.Kind
			}
			error_(pr.TokPos, "//cgo:%s %s: C.%s is %s, not a %s", pr.Kind, declKey(pr), pr.CName, kind, want)
		}
	}
	if nerrors > 0 {
		return
	}

	// Map anonymous struct and union names to typedef names
	// deterministically.
	anonToName := make(map[string]string)
	for _, name := range slices.Sorted(maps.Keys(typedef)) {
		def := typedef[name]
		if isAnonCtype(name) {
			continue
		}
		if id, ok := def.Go.(*ast.Ident); ok && isAnonCtype(id.Name) {
			if _, exists := anonToName[id.Name]; !exists {
				anonToName[id.Name] = strings.TrimPrefix(name, "_Ctype_")
			}
		}
	}

	// Name the C types: types bound by //cgo:type pragmas get the pragma's
	// Go name, and other types get an invented name.
	typeToGoName := make(map[string]string)
	usedGoNames := make(map[string]bool)
	for name, other := range otherDecls {
		if other && !strings.Contains(name, ".") {
			usedGoNames[name] = true
		}
	}
	pragmaType := make(map[string]bool) // _Ctype_ names bound by pragmas
	for _, pr := range allPragmas {
		if pr.RecvType == "" {
			usedGoNames[pr.GoName] = true
		}
		if pr.Kind != "type" {
			continue
		}
		if id, ok := p.Name[pr.CName].Type.Go.(*ast.Ident); ok {
			names := []string{id.Name}
			if typedefName, ok := anonToName[id.Name]; ok {
				names = append(names, "_Ctype_"+typedefName)
			}
			for _, name := range names {
				typeToGoName[name] = pr.GoName
				typeToGoName[strings.TrimPrefix(name, "_Ctype_")] = pr.GoName
				pragmaType[name] = true
			}
		}
	}
	cOrdinaryNames := make(map[string]bool)
	for _, key := range nameKeys(p.Name) {
		n := p.Name[key]
		if n.FuncType != nil || n.IsVar() || n.Const != "" {
			cOrdinaryNames[n.C] = true
		}
	}
	var typeKeys []string
	for name := range typedef {
		if _, ok := primitiveCtypeToGo(name); ok || pragmaType[name] {
			continue
		}
		if isAnonCtype(name) {
			// Anonymous types are written as their typedef name, if
			// any, or inline (see formatGoBindingTypeExpr).
			continue
		}
		typeKeys = append(typeKeys, name)
	}
	slices.Sort(typeKeys)
	for _, name := range typeKeys {
		cleanName := strings.TrimPrefix(name, "_Ctype_")
		goName := "_" + cleanName
		if tag, ok := strings.CutPrefix(cleanName, "struct_"); ok {
			if typedef["_Ctype_"+tag] == nil && !cOrdinaryNames[tag] && !usedGoNames["_"+tag] {
				goName = "_" + tag
			}
		}
		for usedGoNames[goName] {
			goName += "_"
		}
		usedGoNames[goName] = true
		typeToGoName[name] = goName
		typeToGoName[cleanName] = goName
	}
	goType := func(expr ast.Expr) string {
		return p.formatGoBindingTypeExpr(expr, anonToName, typeToGoName)
	}

	// typeDecl returns the declaration of Go type goName for the C type
	// _Ctype_ name.
	typeDecl := func(goName, name string) string {
		def := typedef[name]
		if def == nil {
			if prim, ok := primitiveCtypeToGo(name); ok {
				return fmt.Sprintf("type %s = %s", goName, prim)
			}
			return fmt.Sprintf("type %s = %s", goName, name)
		}
		actualDef := def.Go
		isAlias := false
		if id, ok := actualDef.(*ast.Ident); ok {
			if targetDef, ok := typedef[id.Name]; ok && isAnonCtype(id.Name) {
				actualDef = targetDef.Go
			} else if strings.HasPrefix(id.Name, "_Ctype_") && !isAnonCtype(id.Name) {
				isAlias = true
			}
		}
		if isAlias {
			return fmt.Sprintf("type %s = %s", goName, goType(actualDef))
		}
		return fmt.Sprintf("type %s %s", goName, goType(actualDef))
	}

	var pragmas, decls bytes.Buffer
	for _, pr := range allPragmas {
		pragmas.WriteString(pr.Text + "\n")
		if decl, ok := userDecls[declKey(pr)]; ok {
			// Keep the input's own declaration.
			if !otherDecls[declKey(pr)] {
				decls.WriteString(decl + "\n\n")
			}
			continue
		}
		recv := ""
		if pr.RecvType != "" {
			recv = "(*" + pr.RecvType + ") "
		}
		n := p.Name[pr.CName]
		switch pr.Kind {
		case "dynload":
			fmt.Fprintf(&decls, "type %s struct {\n\tcgo.Library\n}\n\n", pr.GoName)
		case "type":
			id, _ := n.Type.Go.(*ast.Ident)
			if id == nil {
				fmt.Fprintf(&decls, "type %s %s\n\n", pr.GoName, goType(n.Type.Go))
				break
			}
			fmt.Fprintf(&decls, "%s\n\n", typeDecl(pr.GoName, id.Name))
		case "const":
			fmt.Fprintf(&decls, "const %s = %s\n\n", pr.GoName, n.Const)
		case "var":
			if recv != "" {
				fmt.Fprintf(&decls, "func %s%s() *%s\n\n", recv, pr.GoName, goType(n.Type.Go))
				break
			}
			fmt.Fprintf(&decls, "var %s %s\n\n", pr.GoName, goType(n.Type.Go))
		case "func":
			var params []string
			for i, t := range n.FuncType.Params {
				params = append(params, fmt.Sprintf("p%d %s", i, goType(t.Go)))
			}
			ret := ""
			if n.FuncType.Result != nil {
				ret = " " + goType(n.FuncType.Result.Go)
			}
			fmt.Fprintf(&decls, "func %s%s(%s)%s\n\n", recv, pr.GoName, strings.Join(params, ", "), ret)
		}
	}
	// Bind the C types that the declarations depend on.
	var depPragmas bytes.Buffer
	for _, name := range typeKeys {
		goName := typeToGoName[name]
		fmt.Fprintf(&depPragmas, "//cgo:type %s C.%s\n", goName, strings.TrimPrefix(name, "_Ctype_"))
		fmt.Fprintf(&decls, "%s\n\n", typeDecl(goName, name))
	}

	// The pragmas for the C types the declarations depend on follow the
	// input's pragmas, so that running -gen-binding on its own output
	// reproduces it.
	var body bytes.Buffer
	body.Write(pragmas.Bytes())
	body.Write(depPragmas.Bytes())
	body.WriteString("\n")
	body.Write(decls.Bytes())

	// The C preambles of the input files, without any pragmas in them.
	// Files with identical preambles contribute them once.
	var preamble bytes.Buffer
	seenPreamble := make(map[string]bool)
	for _, f := range p.genFiles {
		var buf bytes.Buffer
		forEachImportC(f.gen.ast, func(_ *ast.GenDecl, _ *ast.ImportSpec, doc *ast.CommentGroup) {
			if doc == nil || buf.Len() > 0 {
				return
			}
			for _, c := range doc.List {
				if !isCgoPragmaComment(c.Text) {
					buf.WriteString(c.Text + "\n")
				}
			}
		})
		if text := buf.String(); text != "" && !seenPreamble[text] {
			seenPreamble[text] = true
			preamble.WriteString(text)
		}
	}

	// Imports used by the declarations (detected from parsed declaration AST,
	// ignoring comments, string literals, and //cgo: pragmas).
	usedPkgs := make(map[string]bool)
	if declAST, err := parser.ParseFile(token.NewFileSet(), "", "package "+pkgName.Name+"\n"+decls.String(), 0); err == nil {
		ast.Inspect(declAST, func(n ast.Node) bool {
			if sel, ok := n.(*ast.SelectorExpr); ok {
				if id, ok := sel.X.(*ast.Ident); ok {
					usedPkgs[id.Name] = true
				}
			}
			return true
		})
	}
	var imports []string
	importedPaths := make(map[string]string) // by package name
	for _, f := range p.genFiles {
		g := f.gen
		for _, imp := range g.ast.Imports {
			path, _ := strconv.Unquote(imp.Path.Value)
			if path == "C" {
				continue
			}
			name := path[strings.LastIndex(path, "/")+1:]
			if imp.Name != nil {
				name = imp.Name.Name
			}
			if name == "_" || name == "." || !usedPkgs[name] {
				continue
			}
			if prev, ok := importedPaths[name]; ok {
				if prev != path {
					error_(imp.Pos(), "-gen-binding: %s refers to both %q and %q in the input files", name, prev, path)
				}
				continue
			}
			imports = append(imports, strings.TrimSpace(string(g.src[fset.Position(imp.Pos()).Offset:fset.Position(imp.End()).Offset])))
			importedPaths[name] = path
		}
	}
	for _, std := range []string{"runtime/cgo", "structs", "unsafe"} {
		name := std[strings.LastIndex(std, "/")+1:]
		if _, ok := importedPaths[name]; usedPkgs[name] && !ok {
			imports = append(imports, strconv.Quote(std))
			importedPaths[name] = std
		}
	}

	var out bytes.Buffer
	fmt.Fprintf(&out, "// Code generated by cmd/cgo -gen-binding; DO NOT EDIT.\n\n")
	fmt.Fprintf(&out, "//go:build %s && %s\n\n", goos, goarch)
	fmt.Fprintf(&out, "package %s\n\n", pkgName.Name)
	out.Write(preamble.Bytes())
	fmt.Fprintf(&out, "import \"C\"\n\n")
	for _, imp := range imports {
		fmt.Fprintf(&out, "import %s\n", imp)
	}
	out.WriteString("\n")
	out.Write(body.Bytes())

	formatted, err := format.Source(out.Bytes())
	if err != nil {
		fatalf("-gen-binding: formatting output: %v\n%s", err, out.Bytes())
	}
	w.Write(formatted)
}

func (p *Package) formatGoBindingTypeExpr(expr ast.Expr, anonToName, typeToGoName map[string]string) string {
	switch e := expr.(type) {
	case *ast.Ident:
		name := e.Name
		if name == "_cgopackage.Incomplete" {
			return "cgo.Incomplete"
		}
		if goType, ok := primitiveCtypeToGo(name); ok {
			return goType
		}
		if typedefName, ok := anonToName[name]; ok {
			if gName, ok := typeToGoName[typedefName]; ok {
				return gName
			}
			return "_" + typedefName
		}
		if def := typedef[name]; def != nil && isAnonCtype(name) {
			// An anonymous struct or union without a typedef name.
			// Write it inline: untagged C types cannot refer to
			// themselves, so this terminates.
			return p.formatGoBindingTypeExpr(def.Go, anonToName, typeToGoName)
		}
		if gName, ok := typeToGoName[name]; ok {
			return gName
		}
		clean := strings.TrimPrefix(name, "_Ctype_")
		if gName, ok := typeToGoName[clean]; ok {
			return gName
		}
		if strings.HasPrefix(name, "_Ctype_") {
			return "_" + clean
		}
		return name
	case *ast.StarExpr:
		return "*" + p.formatGoBindingTypeExpr(e.X, anonToName, typeToGoName)
	case *ast.ArrayType:
		lenStr := ""
		if e.Len != nil {
			var lenBuf bytes.Buffer
			noSourceConf.Fprint(&lenBuf, p.fset, e.Len)
			lenStr = lenBuf.String()
		}
		return "[" + lenStr + "]" + p.formatGoBindingTypeExpr(e.Elt, anonToName, typeToGoName)
	case *ast.StructType:
		var buf bytes.Buffer
		buf.WriteString("struct {\n\t_ structs.HostLayout\n")
		for _, fld := range e.Fields.List {
			typeStr := p.formatGoBindingTypeExpr(fld.Type, anonToName, typeToGoName)
			var names []string
			for _, id := range fld.Names {
				names = append(names, id.Name)
			}
			nameStr := strings.Join(names, ", ")
			if nameStr == "" {
				fmt.Fprintf(&buf, "\t%s\n", typeStr)
			} else {
				fmt.Fprintf(&buf, "\t%s %s\n", nameStr, typeStr)
			}
		}
		buf.WriteString("}")
		return buf.String()
	default:
		var buf bytes.Buffer
		noSourceConf.Fprint(&buf, p.fset, expr)
		return strings.ReplaceAll(buf.String(), "_cgopackage.Incomplete", "cgo.Incomplete")
	}
}
