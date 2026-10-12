package gox

import (
	"go/ast"
	"go/parser"
	"go/token"
	"sort"
	"strconv"
	"strings"
)

// printer.SourcePos can put a //line directive in the same comment group as a
// cgo preamble. cgo passes that comment text to C, where "line /path" is invalid.
// Separate only those directives from the preamble and compensate their line
// number for the added blank line. Never edit user C text.
func protectCgoPreambles(src []byte) []byte {
	fs := token.NewFileSet()
	f, e := parser.ParseFile(fs, "generated.go", src, parser.ParseComments)
	if e != nil {
		return src
	}
	type edit struct {
		start, end int
		text       string
	}
	var edits []edit
	seen := map[token.Pos]bool{}
	for _, decl := range f.Decls {
		g, ok := decl.(*ast.GenDecl)
		if !ok || g.Tok != token.IMPORT {
			continue
		}
		for _, spec := range g.Specs {
			i := spec.(*ast.ImportSpec)
			if i.Path.Value != `"C"` {
				continue
			}
			group := i.Doc
			if group == nil {
				group = g.Doc
			}
			if group == nil {
				continue
			}
			for _, c := range group.List {
				if !strings.HasPrefix(c.Text, "//line ") || seen[c.Pos()] {
					continue
				}
				seen[c.Pos()] = true
				j := strings.LastIndexByte(c.Text, ':')
				n, e := strconv.Atoi(c.Text[j+1:])
				if e != nil || n < 2 {
					continue
				}
				edits = append(edits, edit{fs.Position(c.Pos()).Offset, fs.Position(c.End()).Offset, c.Text[:j+1] + strconv.Itoa(n-1) + "\n"})
			}
		}
	}
	sort.Slice(edits, func(i, j int) bool { return edits[i].start > edits[j].start })
	for _, e := range edits {
		out := append([]byte{}, src[:e.start]...)
		out = append(out, e.text...)
		out = append(out, src[e.end:]...)
		src = out
	}
	return src
}
