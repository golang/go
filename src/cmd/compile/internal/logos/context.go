// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"flag"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
)

// ContextLineCap bounds logos context output lines.
const ContextLineCap = 100

// LogosSymbol names the symbol extracted by -logos.mode=context.
var LogosSymbol string

// init registers the context symbol flag beside its consumer.
func init() {
	flag.StringVar(&LogosSymbol, "logos.symbol", "", "symbol for -logos.mode=context")
}

// UnknownSymbolError reports a symbol missing from a context query.
type UnknownSymbolError struct {
	Pkg    string
	Symbol string
}

// Error formats the unknown symbol report.
func (e *UnknownSymbolError) Error() string {
	return "logos context: " + e.Pkg + " has no symbol " + e.Symbol
}

// ContextLines returns the declaration of symbol, the signatures of its
// methods, and the declarations of package types those mention one level
// deep, with no function bodies, capped at ContextLineCap lines.
func ContextLines(files []*syntax.File, symbol string) ([]string, error) {
	all, types := collectDecls(files)
	roots := matchRoots(all, symbol)
	roots = append(roots, matchMentioners(all, symbol)...)
	roots = append(roots, matchMethods(all, symbol)...)
	if len(roots) == 0 {
		return nil, &UnknownSymbolError{Pkg: contextPkg(files), Symbol: symbol}
	}
	names := map[string]bool{}
	for _, root := range roots {
		collectSigNames(root, names)
	}
	delete(names, symbol)
	var lines []string
	for _, root := range roots {
		lines = append(lines, renderDecl(root)...)
	}
	for _, name := range slices.Sorted(maps.Keys(names)) {
		if decl, ok := types[name]; ok {
			lines = append(lines, renderDecl(decl)...)
		}
	}
	return TruncateLines(lines, ContextLineCap), nil
}

// TruncateLines caps lines at max lines.
func TruncateLines(lines []string, max int) []string {
	if len(lines) <= max {
		return lines
	}
	return lines[:max]
}

// RunContext implements -logos.mode=context: it parses the input files
// without imports and prints the context of -logos.symbol on stdout.
func RunContext() {
	symbol := LogosSymbol
	if symbol == "" {
		fmt.Fprintln(os.Stderr, "logos context: missing -logos.symbol")
		base.Exit(2)
	}
	args := flag.Args()
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "logos context: no input files")
		base.Exit(2)
	}
	files := make([]*syntax.File, 0, len(args))
	failed := false
	for _, arg := range args {
		file, err := syntax.ParseFile(arg, func(err error) { fmt.Fprintln(os.Stderr, err) }, nil, contextFileMode(arg))
		if err != nil {
			failed = true
			continue
		}
		files = append(files, file)
	}
	if failed {
		base.Exit(2)
	}
	lines, err := ContextLines(files, symbol)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		base.Exit(1)
	}
	for _, line := range lines {
		fmt.Println(line)
	}
}

// contextPkg names the package of the first parsed file.
func contextPkg(files []*syntax.File) string {
	if len(files) == 0 {
		return ""
	}
	return files[0].PkgName.Value
}

// contextFileMode selects LogosMode for loGo/s files only.
func contextFileMode(name string) syntax.Mode {
	if contextLogosExt(name) {
		return syntax.LogosMode
	}
	return 0
}

// contextLogosExt reports whether name carries a loGo/s extension.
func contextLogosExt(name string) bool {
	ext := filepath.Ext(name)
	ext = strings.TrimRightFunc(ext, func(r rune) bool { return r == 0xFE0E || r == 0xFE0F })
	switch ext {
	case ".lgo", ".📜", ".⚡", ".ʕ◔ϖ◔ʔ":
		return true
	}
	return false
}

// collectDecls gathers top-level declarations and types by name.
func collectDecls(files []*syntax.File) ([]syntax.Decl, map[string]*syntax.TypeDecl) {
	var all []syntax.Decl
	types := map[string]*syntax.TypeDecl{}
	for _, file := range files {
		for _, decl := range file.DeclList {
			all = append(all, decl)
			if td, ok := decl.(*syntax.TypeDecl); ok {
				if _, dup := types[td.Name.Value]; !dup {
					types[td.Name.Value] = td
				}
			}
		}
	}
	return all, types
}

// matchRoots returns non-method declarations named symbol.
func matchRoots(all []syntax.Decl, symbol string) []syntax.Decl {
	var roots []syntax.Decl
	for _, decl := range all {
		if namedDecl(decl, symbol) {
			roots = append(roots, decl)
		}
	}
	return roots
}

// matchMentioners returns package functions whose signature names symbol.
func matchMentioners(all []syntax.Decl, symbol string) []syntax.Decl {
	var found []syntax.Decl
	for _, decl := range all {
		fd, ok := decl.(*syntax.FuncDecl)
		if !ok || fd.Recv != nil || fd.Name.Value == symbol {
			continue
		}
		names := map[string]bool{}
		collectSigNames(fd, names)
		if names[symbol] {
			found = append(found, decl)
		}
	}
	return found
}

// matchMethods returns methods whose receiver base type is symbol.
func matchMethods(all []syntax.Decl, symbol string) []syntax.Decl {
	var methods []syntax.Decl
	for _, decl := range all {
		fd, ok := decl.(*syntax.FuncDecl)
		if !ok || fd.Recv == nil {
			continue
		}
		if recvBaseType(fd.Recv.Type) == symbol {
			methods = append(methods, decl)
		}
	}
	return methods
}

// namedDecl reports whether decl declares symbol outside a receiver.
func namedDecl(decl syntax.Decl, symbol string) bool {
	switch decl := decl.(type) {
	case *syntax.TypeDecl:
		return decl.Name.Value == symbol
	case *syntax.FuncDecl:
		return decl.Recv == nil && decl.Name.Value == symbol
	case *syntax.VarDecl:
		return nameListHas(decl.NameList, symbol)
	case *syntax.ConstDecl:
		return nameListHas(decl.NameList, symbol)
	}
	return false
}

// nameListHas reports whether names holds symbol.
func nameListHas(names []*syntax.Name, symbol string) bool {
	for _, name := range names {
		if name.Value == symbol {
			return true
		}
	}
	return false
}

// recvBaseType unwraps pointer, paren, and index wrappers to a base name.
func recvBaseType(t syntax.Expr) string {
	switch t := t.(type) {
	case *syntax.Name:
		return t.Value
	case *syntax.Operation:
		if t.Y == nil {
			return recvBaseType(t.X)
		}
	case *syntax.ParenExpr:
		return recvBaseType(t.X)
	case *syntax.IndexExpr:
		return recvBaseType(t.X)
	}
	return ""
}

// collectSigNames gathers type names in the signature of decl.
func collectSigNames(decl syntax.Decl, out map[string]bool) {
	switch decl := decl.(type) {
	case *syntax.TypeDecl:
		fieldTypeNames(decl.TParamList, out)
		typeNames(decl.Type, out)
	case *syntax.FuncDecl:
		if decl.Recv != nil {
			typeNames(decl.Recv.Type, out)
		}
		fieldTypeNames(decl.TParamList, out)
		fieldTypeNames(decl.Type.ParamList, out)
		fieldTypeNames(decl.Type.ResultList, out)
	case *syntax.VarDecl:
		if decl.Type != nil {
			typeNames(decl.Type, out)
		}
	case *syntax.ConstDecl:
		if decl.Type != nil {
			typeNames(decl.Type, out)
		}
	}
}

// fieldTypeNames gathers names in the types of fields.
func fieldTypeNames(fields []*syntax.Field, out map[string]bool) {
	for _, field := range fields {
		if field.Type != nil {
			typeNames(field.Type, out)
		}
	}
}

// typeNames gathers names in composite type expressions.
func typeNames(t syntax.Expr, out map[string]bool) {
	switch t := t.(type) {
	case *syntax.Name:
		out[t.Value] = true
	case *syntax.Operation:
		typeNames(t.X, out)
		if t.Y != nil {
			typeNames(t.Y, out)
		}
	case *syntax.MapType:
		typeNames(t.Key, out)
		typeNames(t.Value, out)
	case *syntax.FuncType:
		fieldTypeNames(t.ParamList, out)
		fieldTypeNames(t.ResultList, out)
	case *syntax.StructType:
		fieldTypeNames(t.FieldList, out)
	case *syntax.InterfaceType:
		fieldTypeNames(t.MethodList, out)
	case *syntax.IndexExpr:
		typeNames(t.X, out)
		typeNames(t.Index, out)
	default:
		typeNamesUnary(t, out)
	}
}

// typeNamesUnary gathers names in single-element type wrappers.
func typeNamesUnary(t syntax.Expr, out map[string]bool) {
	switch t := t.(type) {
	case *syntax.ParenExpr:
		typeNames(t.X, out)
	case *syntax.ArrayType:
		typeNames(t.Elem, out)
	case *syntax.SliceType:
		typeNames(t.Elem, out)
	case *syntax.DotsType:
		typeNames(t.Elem, out)
	case *syntax.ChanType:
		typeNames(t.Elem, out)
	case *syntax.OptionType:
		typeNames(t.Elem, out)
	case *syntax.ListExpr:
		for _, elem := range t.ElemList {
			typeNames(elem, out)
		}
	}
}

// renderDecl prints decl without function bodies as source lines.
func renderDecl(decl syntax.Decl) []string {
	if fd, ok := decl.(*syntax.FuncDecl); ok {
		bodiless := *fd
		bodiless.Body = nil
		decl = &bodiless
	}
	var b strings.Builder
	if _, err := syntax.Fprint(&b, decl, 0); err != nil {
		return nil
	}
	return strings.Split(strings.TrimRight(b.String(), "\n"), "\n")
}
