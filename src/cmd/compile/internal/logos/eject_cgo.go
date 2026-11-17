// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"

	"cmd/compile/internal/syntax"
)

func hasCImport(file *syntax.File) bool {
	for _, decl := range file.DeclList {
		if imp, ok := decl.(*syntax.ImportDecl); ok && imp.Path.Value == `"C"` {
			return true
		}
	}
	return false
}

// ejectSourceHeader keeps package positions on the physical Go file:
// upstream cgocall uses them to reopen every file of a cgo package.
// It also keeps generated line directives out of C imports and their
// preambles, where they could become C text or change the selected doc.
// Original positions resume at the next ordinary declaration. Only the
// selected preamble's active fastcall markers receive source edits.
func ejectSourceHeader(file *syntax.File, data []byte) (map[int]bool, []Edit, error) {
	fs := token.NewFileSet()
	parsed, err := parser.ParseFile(fs, file.Pos().FileBase().Filename(), data, parser.ImportsOnly|parser.ParseComments)
	if err != nil {
		return nil, nil, fmt.Errorf("parse eject source header: %w", err)
	}
	lines := make(map[int]bool)
	headerEnd := parsed.Name.End()
	for line := 1; line <= fs.PositionFor(headerEnd, false).Line; line++ {
		lines[line] = true
	}
	var edits []Edit
	for _, decl := range parsed.Decls {
		gen := decl.(*ast.GenDecl) // ImportsOnly returns only imports.
		headerEnd = gen.End()
		for _, spec := range gen.Specs {
			imp := spec.(*ast.ImportSpec)
			if imp.Path.Value != `"C"` {
				continue
			}
			start := gen.Pos()
			if gen.Doc != nil {
				start = gen.Doc.Pos()
			}
			if imp.Doc != nil && imp.Doc.Pos() < start {
				start = imp.Doc.Pos()
			}
			first := fs.PositionFor(start, false).Line
			last := fs.PositionFor(gen.End(), false).Line
			for line := first; line <= last; line++ {
				lines[line] = true
			}
			doc := imp.Doc
			if doc == nil && len(gen.Specs) == 1 {
				doc = gen.Doc
			}
			if doc != nil {
				edits = append(edits, ejectCgoFastcallEdits(fs, doc, data, file.Pos().FileBase())...)
			}
		}
	}
	// A declaration may share a physical line with the package clause
	// or C import. Resume after the imports, leaving preamble attachment
	// intact while retaining that declaration's position after gofmt.
	start := fs.PositionFor(headerEnd, false).Offset
	scanSegment(data, start, len(data), func(t tok) bool {
		if t.comment || t.kind == token.SEMICOLON {
			return true
		}
		pos := fs.PositionFor(fs.File(parsed.Package).Pos(t.start), false)
		if lines[pos.Line] {
			base := file.Pos().FileBase()
			at := syntax.MakePos(base, uint(pos.Line), uint(pos.Column))
			text := fmt.Sprintf("\n//line %s:%d:%d\n", filepath.Base(base.Filename()), pos.Line, pos.Column)
			edits = append(edits, Edit{Start: at, End: at, Text: text})
		}
		return false
	})
	return lines, edits, nil
}

// cgo concatenates comment bodies and removes carriage returns. Track
// each resulting byte back to the source so split markers have exactly
// the native meaning, without changing unrelated C text or delimiters.
func ejectCgoFastcallEdits(fs *token.FileSet, doc *ast.CommentGroup, data []byte, base *syntax.PosBase) []Edit {
	type sourceByte struct {
		offset int
		line   bool
	}
	var text strings.Builder
	var positions []sourceByte
	for _, comment := range doc.List {
		start := fs.PositionFor(comment.Pos(), false).Offset + 2
		line := data[start-1] == '/'
		end := len(data)
		if line {
			if i := bytes.IndexByte(data[start:], '\n'); i >= 0 {
				end = start + i
			}
		} else {
			end = start + bytes.Index(data[start:], []byte("*/"))
		}
		for i := start; i < end; i++ {
			if data[i] != '\r' {
				text.WriteByte(data[i])
				positions = append(positions, sourceByte{i, line})
			}
		}
		if line {
			text.WriteByte('\n')
			positions = append(positions, sourceByte{end, true})
		}
	}
	var edits []splice
	start := 0
	for line := range strings.SplitSeq(text.String(), "\n") {
		fields := strings.Fields(line)
		if len(fields) == 2 && fields[0] == "//logos:cgo_fastcall" {
			first := positions[start]
			last := positions[start+len(line)-1]
			sep := "\n"
			if first.line {
				sep += "// "
			}
			replacement := "#cgo noescape " + fields[1] + sep + "#cgo nocallback " + fields[1]
			if !first.line && last.line {
				// The replacement joins a block and a line comment.
				// Retain the latter's terminating preamble newline.
				replacement += "\n*/"
			}
			edits = append(edits, splice{start: first.offset, end: last.offset + 1, text: replacement})
		}
		start += len(line) + 1
	}
	return editsOf(data, base, edits)
}
