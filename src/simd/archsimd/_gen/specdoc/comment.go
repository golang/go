// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specdoc

import (
	"go/ast"
	"simd/archsimd/_gen/specgen"
	"strings"
)

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
