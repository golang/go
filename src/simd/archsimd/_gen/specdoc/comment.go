// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specdoc

import (
	"go/ast"
	"go/token"
	"path"
	"simd/archsimd/_gen/specgen"
	"slices"
	"sort"
	"strconv"
	"strings"
)

// emulationDependencies returns the public SIMD operations called directly by
// fd's implementation, in source order. Conversion operations are omitted
// because they don't contribute to the cost of an emulation.
func emulationDependencies(fd *ast.FuncDecl, imports map[string]bool) []string {
	if fd.Body == nil {
		return nil
	}

	type dependency struct {
		name string
		pos  token.Pos
	}
	var calls []dependency
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}

		var name string
		var pos token.Pos
		switch fun := call.Fun.(type) {
		case *ast.Ident:
			name = fun.Name
			pos = fun.Pos()
		case *ast.SelectorExpr:
			if id, ok := fun.X.(*ast.Ident); ok {
				// Imported package calls and feature queries such as
				// X86.AVX512BITALG are not SIMD API operations.
				if imports[id.Name] || ast.IsExported(id.Name) {
					return true
				}
			}
			name = fun.Sel.Name
			pos = fun.Sel.Pos()
		}
		if !ast.IsExported(name) || isNoOpConversion(name) {
			return true
		}
		calls = append(calls, dependency{name, pos})
		return true
	})
	sort.Slice(calls, func(i, j int) bool { return calls[i].pos < calls[j].pos })

	seen := make(map[string]bool)
	var deps []string
	for _, call := range calls {
		if !seen[call.name] {
			seen[call.name] = true
			deps = append(deps, call.name)
		}
	}
	return deps
}

func isNoOpConversion(name string) bool {
	return strings.HasPrefix(name, "As") ||
		strings.HasPrefix(name, "ToBits") ||
		strings.HasPrefix(name, "BitsTo") ||
		strings.HasPrefix(name, "ReshapeTo") ||
		strings.HasPrefix(name, "ToInt") ||
		strings.HasPrefix(name, "ToUint") ||
		strings.HasPrefix(name, "ToFloat")
}

// fillEmulationNote replaces a bare Emulated marker with the direct public
// operations used by fd. It preserves any CPU Feature suffix.
func fillEmulationNote(doc []specgen.Paragraph, fd *ast.FuncDecl, imports map[string]bool) []specgen.Paragraph {
	deps := emulationDependencies(fd, imports)
	if len(deps) == 0 {
		return doc
	}

	out := append([]specgen.Paragraph(nil), doc...)
	for i := range out {
		if out[i].Kind != specgen.ImplementationNote ||
			!(out[i].Text == "Emulated" || strings.HasPrefix(out[i].Text, "Emulated,") || strings.HasPrefix(out[i].Text, "Emulated:")) {
			continue
		}
		feature := ""
		if j := strings.Index(out[i].Text, ", CPU Feature:"); j >= 0 {
			feature = strings.TrimPrefix(out[i].Text[j:], ", ")
		} else if j := strings.Index(out[i].Text, ", CPU Feature "); j >= 0 {
			// Normalize the handful of notes missing the colon after Feature.
			feature = "CPU Feature: " + strings.TrimPrefix(out[i].Text[j:], ", CPU Feature ")
		}
		out[i].Text = "Emulated: " + strings.Join(deps, ", ")
		if feature != "" {
			out = slices.Insert(out, i+1, specgen.Paragraph{Kind: specgen.ImplementationNote, Text: feature})
		}
		break
	}
	return out
}

func importNames(file *ast.File) map[string]bool {
	names := make(map[string]bool)
	for _, imp := range file.Imports {
		if imp.Name != nil {
			if imp.Name.Name != "_" && imp.Name.Name != "." {
				names[imp.Name.Name] = true
			}
			continue
		}
		importPath, err := strconv.Unquote(imp.Path.Value)
		if err == nil {
			names[path.Base(importPath)] = true
		}
	}
	return names
}

// parseCommentGroup splits an AST comment group into classified paragraphs.
func parseCommentGroup(cg *ast.CommentGroup) []specgen.Paragraph {
	if cg == nil || len(cg.List) == 0 {
		return nil
	}

	var paras []specgen.Paragraph
	var para strings.Builder
	flush := func() {
		if para.Len() == 0 {
			return
		}
		text := para.String()
		para.Reset()

		kind := specgen.SpecOwned
		if specgen.HasNotePrefix(text) {
			kind = specgen.ImplementationNote
		}
		paras = append(paras, specgen.Paragraph{Kind: kind, Text: text})
	}
	for _, c := range cg.List {
		if isDirectiveLine(c.Text) {
			flush()
			paras = append(paras, specgen.Paragraph{Kind: specgen.DirectiveComment, Text: c.Text[2:]})
			continue
		}

		// Drop leading "//" or "// ". If it's "//\t", we keep the "\t"
		text, ok := strings.CutPrefix(c.Text, "// ")
		if !ok {
			text, ok = strings.CutPrefix(text, "//")
			if !ok {
				panic("comment does not start with //?")
			}
		}
		if text == "" {
			// Paragraph break
			flush()
			continue
		}
		if para.Len() > 0 {
			para.WriteByte('\n')
		}
		para.WriteString(text)
	}
	flush()

	return paras
}

// isDirectiveLine reports whether comment line is a compiler/tool directive
// comment.
func isDirectiveLine(line string) bool {
	line = strings.TrimSpace(line)
	if _, ok := ast.ParseDirective(0, line); ok {
		return true
	}
	if strings.HasPrefix(line, "//line ") {
		return true
	}
	return false
}
