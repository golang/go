// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"flag"
	"fmt"
	"internal/logosfile"
	"maps"
	"os"
	"slices"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
)

const ContextLineCap = 100

var LogosSymbol string

func init() {
	flag.StringVar(&LogosSymbol, "logos.symbol", "", "symbol for -logos.mode=context")
}

type UnknownSymbolError struct {
	Pkg    string
	Symbol string
}

func (e *UnknownSymbolError) Error() string {
	return "logos context: " + e.Pkg + " has no symbol " + e.Symbol
}

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
	if len(lines) > ContextLineCap {
		lines = lines[:ContextLineCap]
	}
	return lines, nil
}

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

func contextPkg(files []*syntax.File) string {
	if len(files) == 0 {
		return ""
	}
	return files[0].PkgName.Value
}

func contextFileMode(name string) syntax.Mode {
	if logosfile.Is(name) {
		return syntax.LogosMode
	}
	return 0
}

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

func matchRoots(all []syntax.Decl, symbol string) []syntax.Decl {
	var roots []syntax.Decl
	for _, decl := range all {
		if namedDecl(decl, symbol) {
			roots = append(roots, decl)
		}
	}
	return roots
}

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

func nameListHas(names []*syntax.Name, symbol string) bool {
	for _, name := range names {
		if name.Value == symbol {
			return true
		}
	}
	return false
}

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

func fieldTypeNames(fields []*syntax.Field, out map[string]bool) {
	for _, field := range fields {
		if field.Type != nil {
			typeNames(field.Type, out)
		}
	}
}

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
