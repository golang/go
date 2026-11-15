// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/link/internal/loader"
	"go/ast"
	"go/parser"
	"go/token"
	"internal/testenv"
	"maps"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestLogosCapRawEntriesComplete(t *testing.T) {
	src := filepath.Join(testenv.GOROOT(t), "src")
	fset := token.NewFileSet()
	syscallFiles := logosCapParseDir(t, fset, filepath.Join(src, "syscall"))
	runtimeFiles := logosCapParseDir(t, fset, filepath.Join(src, "runtime"))
	pushed := logosCapPushed(slices.Concat(syscallFiles, runtimeFiles))
	entries := logosCapRawEntries(fset, syscallFiles, pushed)
	if len(entries) == 0 {
		t.Fatal("no raw entry point found in GOROOT/src/syscall")
	}
	for _, name := range slices.Sorted(maps.Keys(entries)) {
		sink := "syscall." + name
		if capabilitySinks[sink] != CapSyscall && !loader.LinknameBlocked("example.com/dep", sink) {
			t.Errorf("%s (%s) is a raw entry point, but not a CapSyscall sink and not blocked by the linker", sink, entries[name])
		}
	}
	for sink, kind := range capabilitySinks {
		name, ok := strings.CutPrefix(sink, "syscall.")
		if _, found := entries[name]; kind == CapSyscall && ok && !found {
			t.Errorf("sink %q is no raw entry point of package syscall", sink)
		}
	}
}

func logosCapParseDir(t *testing.T, fset *token.FileSet, dir string) []*ast.File {
	t.Helper()
	names, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil {
		t.Fatal(err)
	}
	var files []*ast.File
	for _, name := range names {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := parser.ParseFile(fset, name, nil, parser.ParseComments)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, f)
	}
	return files
}

func logosCapPushed(files []*ast.File) map[string]bool {
	pushed := make(map[string]bool)
	for _, f := range files {
		for _, fields := range logosCapLinknames(f) {
			if f.Name.Name == "syscall" && len(fields) == 1 {
				pushed[fields[0]] = true
			}
			if name, ok := logosCapSyscallTarget(fields); ok {
				pushed[name] = true
			}
		}
	}
	return pushed
}

func logosCapLinknames(f *ast.File) [][]string {
	var out [][]string
	for _, group := range f.Comments {
		for _, c := range group.List {
			if rest, ok := strings.CutPrefix(c.Text, "//go:linkname "); ok {
				out = append(out, strings.Fields(rest))
			}
		}
	}
	return out
}

func logosCapSyscallTarget(fields []string) (string, bool) {
	if len(fields) != 2 {
		return "", false
	}
	return strings.CutPrefix(fields[1], "syscall.")
}

func logosCapRawEntries(fset *token.FileSet, files []*ast.File, pushed map[string]bool) map[string]string {
	entries := make(map[string]string)
	for _, f := range files {
		if f.Name.Name != "syscall" {
			continue
		}
		for _, decl := range f.Decls {
			fd, ok := decl.(*ast.FuncDecl)
			if !ok || !logosCapIsRaw(fd) || !(ast.IsExported(fd.Name.Name) || pushed[fd.Name.Name]) {
				continue
			}
			name := logosCapEntryName(fd)
			entries[name] = filepath.Base(fset.File(fd.Pos()).Name())
		}
	}
	return entries
}

func logosCapEntryName(fd *ast.FuncDecl) string {
	if fd.Recv == nil {
		return fd.Name.Name
	}
	switch recv := fd.Recv.List[0].Type.(type) {
	case *ast.StarExpr:
		return "(*" + recv.X.(*ast.Ident).Name + ")." + fd.Name.Name
	default:
		return recv.(*ast.Ident).Name + "." + fd.Name.Name
	}
}

func logosCapIsRaw(fd *ast.FuncDecl) bool {
	if fd.Type.Params == nil || fd.Type.Results == nil || len(fd.Type.Params.List) == 0 {
		return false
	}
	for _, p := range fd.Type.Params.List {
		if !logosCapIsUintptr(p.Type) {
			return false
		}
	}
	results := logosCapResultTypes(fd.Type.Results)
	return len(results) >= 2 && logosCapIsUintptr(results[0]) && logosCapIsUintptr(results[1])
}

func logosCapResultTypes(list *ast.FieldList) []ast.Expr {
	var out []ast.Expr
	for _, f := range list.List {
		for range max(len(f.Names), 1) {
			out = append(out, f.Type)
		}
	}
	return out
}

func logosCapIsUintptr(e ast.Expr) bool {
	if ellipsis, ok := e.(*ast.Ellipsis); ok {
		e = ellipsis.Elt
	}
	id, ok := e.(*ast.Ident)
	return ok && id.Name == "uintptr"
}
