// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"os"
	"path/filepath"
	"strings"
)

// ExportFunc describes an //export Go function callable from C in a cgo
// binding package.
type ExportFunc struct {
	Name   string
	Params []string
	Ret    string
	Func   *ast.FuncDecl
	Imp    fileImports
}

// rawBindingExport records an //export function declaration along with the
// import aliases of the source file it was declared in.
type rawBindingExport struct {
	fn  *ast.FuncDecl
	imp fileImports
}

// collectBindingExports returns the //export functions declared in file,
// reporting errors for exports that cannot be supported without a C
// compiler. expMap records the export names seen so far.
func (p *Package) collectBindingExports(file *ast.File, expMap map[string]bool) []rawBindingExport {
	var rawExports []rawBindingExport
	imp := collectFileImports(file)
	for _, decl := range file.Decls {
		d, ok := decl.(*ast.FuncDecl)
		if !ok || d.Doc == nil {
			continue
		}
		for _, c := range d.Doc.List {
			trimmed := strings.TrimSpace(c.Text)
			after, hasExport := strings.CutPrefix(trimmed, "//export")
			if !hasExport || (after != "" && after[0] != ' ' && after[0] != '\t') {
				continue
			}
			expName := strings.TrimSpace(after)
			if expName == "" || strings.ContainsAny(expName, " \t") {
				error_(c.Pos(), "malformed //export directive: %s", trimmed)
				continue
			}
			if expName != d.Name.Name {
				error_(c.Pos(), "export comment has wrong name %q, want %q", expName, d.Name.Name)
				continue
			}
			if d.Recv != nil {
				error_(d.Pos(), "cannot export method %s: //export functions cannot have a receiver", d.Name.Name)
				continue
			}
			if d.Type.TypeParams != nil && len(d.Type.TypeParams.List) > 0 {
				error_(d.Pos(), "cannot export generic function %s", d.Name.Name)
				continue
			}
			if d.Body == nil {
				error_(d.Pos(), "cannot export bodyless function declaration %s", d.Name.Name)
				continue
			}
			if d.Type.Results != nil && d.Type.Results.NumFields() > 1 {
				error_(d.Pos(), "cannot export function %s with multiple return values", d.Name.Name)
				continue
			}
			if expMap[expName] {
				error_(d.Pos(), "duplicate //export name %s", expName)
				continue
			}
			rawExports = append(rawExports, rawBindingExport{fn: d, imp: imp})
			expMap[expName] = true
			break
		}
	}
	return rawExports
}

// exportBindingFunc returns the static BindingFunc used to invoke the C-ABI
// wrapper of an //export Go function from Go code (via C.<name>(...)).
func exportBindingFunc(ef ExportFunc) BindingFunc {
	return BindingFunc{
		Name:        ef.Name,
		CName:       ef.Name,
		BindC:       true,
		Static:      true,
		Params:      ef.Params,
		Ret:         ef.Ret,
		MayCallback: true,
	}
}

// resolveBindingExports validates and computes the Go argument layouts of the
// exported functions rawExports.
func (p *Package) resolveBindingExports(entry *Bindings, rawExports []rawBindingExport) []ExportFunc {
	boundConflict := make(map[string]bool)
	for _, fn := range entry.Funcs {
		boundConflict[fn.Name] = true
		if !fn.Indirect {
			boundConflict[fn.GoName] = true
			boundConflict[fn.CName] = true
		}
	}
	for _, v := range entry.Vars {
		boundConflict[v.GoName] = true
		boundConflict[v.CName] = true
	}
	for goName, cName := range entry.TypeGoToC {
		boundConflict[goName] = true
		boundConflict[cName] = true
	}
	for cName, goName := range entry.ConstCToGo {
		boundConflict[goName] = true
		boundConflict[cName] = true
	}
	for _, t := range entry.DynloadTypes {
		boundConflict[t] = true
	}

	var expFuncs []ExportFunc
	for _, re := range rawExports {
		fn := re.fn
		if fn.Name.Name == "malloc" || fn.Name.Name == "_cgo_malloc" || builtinDefs[fn.Name.Name] != "" {
			error_(fn.Pos(), "//export %s conflicts with cgo builtin", fn.Name.Name)
			continue
		}
		if boundConflict[fn.Name.Name] {
			error_(fn.Pos(), "//export %s conflicts with //cgo: binding", fn.Name.Name)
			continue
		}
		var params []string
		if fn.Type.Params != nil {
			for _, field := range fn.Type.Params.List {
				if isUnsupportedExportType(field.Type, entry.typeSpecs, make(map[string]bool)) {
					var tb bytes.Buffer
					noSourceConf.Fprint(&tb, p.fset, field.Type)
					error_(field.Type.Pos(), "Go type not supported in //export without a C compiler: %s", tb.String())
					continue
				}
				if entry.resolveUsedTypeInFile != nil {
					entry.resolveUsedTypeInFile(field.Type, false, re.imp)
				}
				pType := p.cgoASTToGoDefInFile(field.Type, entry, re.imp)
				if why := cTypeError(pType, entry.TypeMap); why != "" {
					error_(field.Type.Pos(), "//export %s: parameter type %s cannot be passed from C: %s", fn.Name.Name, pType, why)
					continue
				}
				for range max(1, len(field.Names)) {
					params = append(params, pType)
				}
			}
		}
		ret := ""
		if fn.Type.Results != nil && len(fn.Type.Results.List) > 0 {
			retExpr := fn.Type.Results.List[0].Type
			if isUnsupportedExportType(retExpr, entry.typeSpecs, make(map[string]bool)) {
				var tb bytes.Buffer
				noSourceConf.Fprint(&tb, p.fset, retExpr)
				error_(retExpr.Pos(), "Go type not supported in //export without a C compiler: %s", tb.String())
			} else {
				if entry.resolveUsedTypeInFile != nil {
					entry.resolveUsedTypeInFile(retExpr, false, re.imp)
				}
				ret = p.cgoASTToGoDefInFile(retExpr, entry, re.imp)
				if why := cTypeError(ret, entry.TypeMap); why != "" {
					error_(retExpr.Pos(), "//export %s: result type %s cannot be returned to C: %s", fn.Name.Name, ret, why)
					ret = ""
				}
			}
		}
		expFuncs = append(expFuncs, ExportFunc{
			Name:   fn.Name.Name,
			Params: params,
			Ret:    ret,
			Func:   fn,
			Imp:    re.imp,
		})
	}
	return expFuncs
}

// writeBindingExports emits the Go export directives, _Cfpvar_fp_
// function-pointer variables, and _cgoexp_ crosscall2 wrappers for expFuncs.
func (p *Package) writeBindingExports(f *os.File, expFuncs []ExportFunc, typeMap map[string]BindingType, abiCfg *ABIConfig) {
	for _, ef := range expFuncs {
		fmt.Fprintf(f, "//go:cgo_export_dynamic %s\n", ef.Name)
		fmt.Fprintf(f, "//go:cgo_export_static %s\n\n", ef.Name)
	}
	for _, ef := range expFuncs {
		fmt.Fprintf(f, "//go:linkname __cgofn_%s %s\n", ef.Name, ef.Name)
		fmt.Fprintf(f, "var __cgofn_%s byte\n", ef.Name)
		fmt.Fprintf(f, "var _Cfpvar_fp_%s unsafe.Pointer = unsafe.Pointer(&__cgofn_%s)\n\n", ef.Name, ef.Name)

		if len(ef.Params) > 0 || ef.Ret != "" {
			fmt.Fprintf(f, "type _cgo_arg_%s struct {\n", ef.Name)
			for i, pt := range ef.Params {
				fmt.Fprintf(f, "\tp%d %s\n", i, pt)
			}
			if ef.Ret != "" {
				fmt.Fprintf(f, "\tret %s\n", ef.Ret)
			}
			fmt.Fprintf(f, "}\n\n")
		}

		var callArgs []string
		for i := range ef.Params {
			callArgs = append(callArgs, fmt.Sprintf("args.p%d", i))
		}
		callStr := fmt.Sprintf("%s(%s)", ef.Name, strings.Join(callArgs, ", "))

		fmt.Fprintf(f, "//go:linkname _cgoexp%s_%s _cgoexp%s_%s\n", cPrefix, ef.Name, cPrefix, ef.Name)
		fmt.Fprintf(f, "func _cgoexp%s_%s(a unsafe.Pointer) {\n", cPrefix, ef.Name)
		if len(ef.Params) == 0 && ef.Ret == "" {
			fmt.Fprintf(f, "\t%s()\n", ef.Name)
		} else {
			fmt.Fprintf(f, "\targs := (*_cgo_arg_%s)(a)\n", ef.Name)
			if ef.Ret != "" {
				fmt.Fprintf(f, "\targs.ret = %s\n", callStr)
				if abiCfg.inspectType(ef.Ret, typeMap).hasPointers {
					if ef.Func != nil && p.fset != nil {
						pos := p.fset.Position(ef.Func.Pos())
						fmt.Fprintf(f, "//line %s:%d\n", filepath.ToSlash(pos.Filename), pos.Line)
					}
					fmt.Fprintf(f, "\t_cgoCheckResult(args.ret)\n")
					if ef.Func != nil && p.fset != nil {
						fmt.Fprintf(f, "//line :1\n")
					}
				}
			} else {
				fmt.Fprintf(f, "\t%s\n", callStr)
			}
		}
		fmt.Fprintf(f, "}\n\n")

		fmt.Fprintf(f, "var _cgo_fn_%s = _cgoexp%s_%s\n\n", ef.Name, cPrefix, ef.Name)
	}
}

func isUnsupportedExportType(expr ast.Expr, typeSpecs map[string]*ast.TypeSpec, seen map[string]bool) bool {
	switch t := expr.(type) {
	case *ast.ParenExpr:
		return isUnsupportedExportType(t.X, typeSpecs, seen)
	case *ast.Ellipsis, *ast.MapType, *ast.ChanType, *ast.InterfaceType, *ast.FuncType:
		return true
	case *ast.ArrayType:
		if t.Len == nil {
			return true
		}
		return isUnsupportedExportType(t.Elt, typeSpecs, seen)
	case *ast.StructType:
		if t.Fields != nil {
			for _, fld := range t.Fields.List {
				if isUnsupportedExportType(fld.Type, typeSpecs, seen) {
					return true
				}
			}
		}
		return false
	case *ast.StarExpr:
		if at, ok := t.X.(*ast.ArrayType); ok && at.Len != nil {
			if bl, ok := at.Len.(*ast.BasicLit); ok && bl.Value == "0" {
				return false
			}
		}
		return isUnsupportedExportType(t.X, typeSpecs, seen)
	case *ast.Ident:
		switch t.Name {
		case "string", "complex64", "complex128", "any", "error":
			return true
		}
		if ts, ok := typeSpecs[t.Name]; ok && !seen[t.Name] {
			seen[t.Name] = true
			res := isUnsupportedExportType(ts.Type, typeSpecs, seen)
			delete(seen, t.Name)
			return res
		}
	case *ast.SelectorExpr:
		if id, ok := t.X.(*ast.Ident); ok && id.Name == "C" {
			switch t.Sel.Name {
			case "_GoString_", "_GoBytes_", "complexfloat", "complexdouble":
				return true
			}
		}
	}
	return false
}

func bindingExportCType(expr ast.Expr, entry *Bindings, imp fileImports, seen map[string]bool) (string, bool) {
	switch t := expr.(type) {
	case *ast.ParenExpr:
		return bindingExportCType(t.X, entry, imp, seen)
	case *ast.StarExpr:
		if at, ok := t.X.(*ast.ArrayType); ok && at.Len != nil {
			if bl, ok := at.Len.(*ast.BasicLit); ok && bl.Value == "0" {
				return "void*", true
			}
		}
		if sel, ok := t.X.(*ast.SelectorExpr); ok {
			if id, ok := sel.X.(*ast.Ident); ok && id.Name == "C" && sel.Sel.Name == "void" {
				return "void*", true
			}
		}
		elem, ok := bindingExportCType(t.X, entry, imp, seen)
		if !ok {
			return "", false
		}
		return elem + "*", true
	case *ast.SelectorExpr:
		if id, ok := t.X.(*ast.Ident); ok {
			if imp.unsafe[id.Name] && t.Sel.Name == "Pointer" {
				return "void*", true
			}
			if id.Name == "C" {
				switch t.Sel.Name {
				case "int8_t":
					return "GoInt8", true
				case "uint8_t":
					return "GoUint8", true
				case "int16_t":
					return "GoInt16", true
				case "uint16_t":
					return "GoUint16", true
				case "int32_t":
					return "GoInt32", true
				case "uint32_t":
					return "GoUint32", true
				case "int64_t":
					return "GoInt64", true
				case "uint64_t":
					return "GoUint64", true
				case "uintptr_t":
					return "GoUintptr", true
				case "intptr_t", "ssize_t":
					return "ptrdiff_t", true
				case "bool":
					return "_Bool", true
				default:
					return cname(t.Sel.Name), true
				}
			}
		}
	case *ast.Ident:
		if imp.unsafe["."] && t.Name == "Pointer" {
			return "void*", true
		}
		if cName, ok := entry.TypeGoToAnyC[t.Name]; ok && cName != "" {
			return cname(cName), true
		}
		if t.Name == "uintptr" {
			return "GoUintptr", true
		}
		if gt, ok := goTypes[t.Name]; ok {
			return gt.C.String(), true
		}
		if ts, ok := entry.typeSpecs[t.Name]; ok && !seen[t.Name] {
			seen[t.Name] = true
			res, ok := bindingExportCType(ts.Type, entry, imp, seen)
			delete(seen, t.Name)
			return res, ok
		}
	}
	return "", false
}

func (p *Package) writeBindingExportHeader(headerPath string, entry *Bindings, expFuncs []ExportFunc, sources []bindingSource) {
	var preamble strings.Builder
	for _, bs := range sources {
		if bs.file == nil {
			continue
		}
		forEachImportC(bs.file, func(_ *ast.GenDecl, _ *ast.ImportSpec, doc *ast.CommentGroup) {
			if doc == nil {
				return
			}
			for line := range strings.SplitSeq(commentText(doc), "\n") {
				l := strings.TrimSpace(line)
				if (len(l) >= 5 && l[:4] == "#cgo" && (l[4] == ' ' || l[4] == '\t')) || isCgoPragmaComment("//"+l) {
					preamble.WriteByte('\n')
					continue
				}
				preamble.WriteString(line)
				preamble.WriteByte('\n')
			}
		})
	}
	origPreamble := p.Preamble
	if p.Preamble == "" {
		p.Preamble = preamble.String()
	}
	defer func() { p.Preamble = origPreamble }()

	formatCType := func(expr ast.Expr, imp fileImports) string {
		ct, ok := bindingExportCType(expr, entry, imp, make(map[string]bool))
		if !ok {
			var tb bytes.Buffer
			noSourceConf.Fprint(&tb, p.fset, expr)
			fatalf("%s: cannot represent Go type %s in _cgo_export.h (bind it with //cgo:type)", p.fset.Position(expr.Pos()), tb.String())
		}
		return ct
	}

	var hdr bytes.Buffer
	p.writeExportHeader(&hdr)
	for _, ef := range expFuncs {
		if ef.Func == nil {
			continue
		}
		fntype := ef.Func.Type
		cRet := "void"
		if fntype.Results != nil && len(fntype.Results.List) > 0 {
			cRet = formatCType(fntype.Results.List[0].Type, ef.Imp)
		}
		var s strings.Builder
		fmt.Fprintf(&s, "%s %s(", cRet, ef.Name)
		if fntype.Params != nil && len(fntype.Params.List) > 0 {
			forFieldList(fntype.Params, func(i int, aname string, atype ast.Expr) {
				if i > 0 {
					s.WriteString(", ")
				}
				cParamType := formatCType(atype, ef.Imp)
				fmt.Fprintf(&s, "%s %s", cParamType, exportParamName(aname, i))
			})
		} else {
			s.WriteString("void")
		}
		s.WriteByte(')')
		fmt.Fprintf(&hdr, "extern %s;\n", s.String())
	}
	fmt.Fprintf(&hdr, "%s", gccExportHeaderEpilog)

	if err := os.WriteFile(headerPath, hdr.Bytes(), 0666); err != nil {
		fatalf("%s", err)
	}
}
