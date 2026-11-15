// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmd/internal/logos/editmap"
	"cmd/internal/logosowned"
)

func Eject(dir string, files []*syntax.File, src map[*syntax.File][]byte, edits map[*syntax.File][]Edit) error {
	if err := os.MkdirAll(dir, 0o777); err != nil {
		return err
	}
	for _, file := range files {
		data, ok := src[file]
		if !ok {
			continue
		}
		base := file.Pos().FileBase()
		lowered := edits[file]
		var lineEdits []Edit
		if starts := lineStarts(data); len(starts) > 0 {
			type span struct{ s, e int }
			var covered []span
			for _, e := range lowered {
				s, err := offsetOf(starts, len(data), e.Start)
				if err != nil {
					continue
				}
				t, err := offsetOf(starts, len(data), e.End)
				if err != nil {
					continue
				}
				if t > s {
					covered = append(covered, span{s, t})
				}
			}
			slices.SortFunc(covered, func(a, b span) int { return a.s - b.s })
			baseName := filepath.Base(base.Filename())
			var blocked []span
			scanSegment(data, 0, len(data), func(t tok) bool {
				if t.auto {
					return true
				}
				if t.comment {
					if t.end-t.start >= 2 && data[t.start] == '/' && data[t.start+1] == '*' {
						blocked = append(blocked, span{s: t.start, e: t.end})
					}
					return true
				}
				if t.kind == token.STRING && t.start < t.end && data[t.start] == '`' {
					blocked = append(blocked, span{s: t.start, e: t.end})
				}
				return true
			})
			for line, off := range starts {
				no := line + 1
				in := false
				for _, c := range covered {
					if off >= c.s && off < c.e {
						in = true
						break
					}
					if c.s > off {
						break
					}
				}
				if in {
					continue
				}
				inBlocked := false
				for _, u := range blocked {
					if off > u.s && off < u.e {
						inBlocked = true
						break
					}
					if u.s > off {
						break
					}
				}
				if inBlocked {
					continue
				}
				{
					lineEnd := len(data)
					if line+1 < len(starts) {
						lineEnd = starts[line+1]
					}
					j := off
					for j < lineEnd && (data[j] == ' ' || data[j] == '\t' || data[j] == '\r') {
						j++
					}
					if j >= lineEnd || data[j] == '\n' {
						continue
					}
					if j+1 < lineEnd && data[j] == '/' && data[j+1] == '/' {
						continue
					}
				}
				lineEdits = append(lineEdits, Edit{
					Start: syntax.MakePos(base, uint(no), 1),
					End:   syntax.MakePos(base, uint(no), 1),
					Text:  "//line " + baseName + ":" + strconv.Itoa(no) + ":1\n",
				})
			}
			lowered = append(lineEdits, lowered...)
		}
		out, spans, segments, err := ApplyEditsSegments(data, base, lowered)
		if err != nil {
			return err
		}
		name := filepath.Join(dir, ejectedName(base.Filename()))
		if err := os.WriteFile(name, out, 0o666); err != nil {
			return err
		}
		if err := editmap.Write(name+editmap.Suffix, base.Filename(), len(data), len(out), spans, segments); err != nil {
			return err
		}
	}
	return writeOwnedConstantRequests(dir)
}

func writeOwnedConstantRequests(dir string) error {
	requests, err := OwnedConstantRequests()
	if err != nil {
		return err
	}
	data, err := logosowned.Encode(requests)
	if err != nil {
		return err
	}
	f, err := os.CreateTemp(dir, ".logos-owned-*")
	if err != nil {
		return err
	}
	name := f.Name()
	defer os.Remove(name)
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(name, filepath.Join(dir, logosowned.FileName))
}

func ejectedName(name string) string {
	base := filepath.Base(name)
	return strings.TrimSuffix(base, filepath.Ext(base)) + ".go"
}

func DualUseMessage(name string) string {
	return "union " + name + " is used as a type constraint and as a value type; Go cannot express both"
}

type UnionUse int

const (
	UnionConstraintOnly UnionUse = iota
	UnionValueOnly
	UnionDualUse
)

func KeepUnionDecl(m map[*types2.TypeName]UnionUse, obj *types2.TypeName) bool {
	u, ok := m[obj]
	return ok && u != UnionConstraintOnly
}

func HasUnionTerms(iface *types2.Interface) bool {
	for i := range iface.NumEmbeddeds() {
		if IsUnionTerm(iface.EmbeddedType(i)) {
			return true
		}
	}
	return false
}

func IsUnionTerm(t types2.Type) bool {
	u, ok := t.(*types2.Union)
	return ok && u.Len() > 0
}

func isClosedUnion(iface *types2.Interface) bool {
	closed := false
	for i := range iface.NumEmbeddeds() {
		u, ok := iface.EmbeddedType(i).(*types2.Union)
		if !ok {
			continue
		}
		if u.Len() == 0 {
			continue
		}
		closed = true
		for j := range u.Len() {
			if u.Term(j).Tilde() {
				return false
			}
		}
	}
	return closed
}

type unionDecl struct {
	decl  *syntax.TypeDecl
	iface *syntax.InterfaceType
	file  *syntax.File
}

type unionClass struct {
	modes map[*types2.TypeName]UnionUse
	decls map[*types2.TypeName]*unionDecl
}

func ClassifyDeclaredUnions(info *types2.Info, files []*syntax.File) map[*types2.TypeName]UnionUse {
	return classifyUnions(info, files).modes
}

func classifyUnions(info *types2.Info, files []*syntax.File) *unionClass {
	c := &unionClass{
		modes: make(map[*types2.TypeName]UnionUse),
		decls: make(map[*types2.TypeName]*unionDecl),
	}
	if info == nil {
		return c
	}
	unions := make(map[*types2.TypeName]bool)
	var bareEdges [][2]*types2.TypeName
	barePos := make(map[*syntax.Name]bool)
	for _, file := range files {
		syntax.Inspect(file, func(n syntax.Node) bool {
			td, ok := n.(*syntax.TypeDecl)
			if !ok {
				return true
			}
			obj, ok := info.Defs[td.Name].(*types2.TypeName)
			if !ok || obj == nil {
				return true
			}
			if td.Alias {
				return true
			}
			iface, ok := obj.Type().Underlying().(*types2.Interface)
			if !ok || !HasUnionTerms(iface) {
				return true
			}
			unions[obj] = true
			if it, ok := td.Type.(*syntax.InterfaceType); ok {
				c.decls[obj] = &unionDecl{decl: td, iface: it, file: file}
			} else if target := unionNameRef(info, td.Type); target != nil && target != obj {
				bareEdges = append(bareEdges, [2]*types2.TypeName{obj, target})
				if name := bareRefName(td.Type); name != nil {
					barePos[name] = true
				}
			}
			return true
		})
	}
	if len(unions) == 0 {
		return c
	}

	constraint := make(map[*types2.TypeName]bool)
	value := make(map[*types2.TypeName]bool)
	var edges [][2]*types2.TypeName
	edges = append(edges, bareEdges...)
	constraintPos := make(map[*syntax.Name]bool)
	embedPos := make(map[*syntax.Name]bool)
	note := func(obj *types2.TypeName, dst map[*types2.TypeName]bool) {
		if obj != nil {
			dst[obj] = true
		}
	}
	noteAll := func(n *syntax.Name, pos map[*syntax.Name]bool, dst map[*types2.TypeName]bool) {
		if n == nil {
			return
		}
		pos[n] = true
		note(resolveUnionRef(info.Uses[n]), dst)
	}
	for _, file := range files {
		syntax.Inspect(file, func(n syntax.Node) bool {
			var list []*syntax.Field
			switch n := n.(type) {
			case *syntax.TypeDecl:
				list = n.TParamList
			case *syntax.FuncDecl:
				list = n.TParamList
			default:
				return true
			}
			for _, f := range list {
				if f == nil || f.Type == nil {
					continue
				}
				syntax.Inspect(f.Type, func(m syntax.Node) bool {
					if name, ok := m.(*syntax.Name); ok {
						noteAll(name, constraintPos, constraint)
					}
					return true
				})
			}
			return true
		})
	}
	var walkEmbeddedNames func(e syntax.Expr, mark func(*syntax.Name))
	walkEmbeddedNames = func(e syntax.Expr, mark func(*syntax.Name)) {
		e = unwrapParen(e)
		switch e := e.(type) {
		case *syntax.Name:
			mark(e)
		case *syntax.SelectorExpr:
			mark(e.Sel)
		case *syntax.Operation:
			if e.Op == syntax.Or {
				walkEmbeddedNames(e.X, mark)
				walkEmbeddedNames(e.Y, mark)
			}
		}
	}
	markConstraintName := func(name *syntax.Name) {
		noteAll(name, constraintPos, constraint)
	}
	for _, file := range files {
		syntax.Inspect(file, func(n syntax.Node) bool {
			td, ok := n.(*syntax.TypeDecl)
			if !ok || td.Alias {
				return true
			}
			it, ok := td.Type.(*syntax.InterfaceType)
			if !ok {
				return true
			}
			obj, ok := info.Defs[td.Name].(*types2.TypeName)
			if !ok || obj == nil {
				return true
			}
			generic := len(td.TParamList) != 0
			for _, f := range it.MethodList {
				if f == nil || f.Name != nil || f.Type == nil {
					continue
				}
				if generic {
					walkEmbeddedNames(f.Type, markConstraintName)
				} else {
					from := obj
					walkEmbeddedNames(f.Type, func(name *syntax.Name) {
						if name == nil {
							return
						}
						embedPos[name] = true
						if target := resolveUnionRef(info.Uses[name]); target != nil && unions[target] {
							edges = append(edges, [2]*types2.TypeName{from, target})
						}
					})
				}
			}
			return true
		})
	}

	for _, file := range files {
		syntax.Inspect(file, func(n syntax.Node) bool {
			name, ok := n.(*syntax.Name)
			if !ok || constraintPos[name] || embedPos[name] || barePos[name] {
				return true
			}
			note(resolveUnionRef(info.Uses[name]), value)
			return true
		})
	}

	changed := true
	for changed {
		changed = false
		for _, e := range edges {
			from, to := e[0], e[1]
			if value[from] && !value[to] {
				value[to] = true
				changed = true
			}
			if constraint[from] && !constraint[to] {
				constraint[to] = true
				changed = true
			}
		}
	}
	for obj := range unions {
		switch {
		case value[obj] && constraint[obj]:
			c.modes[obj] = UnionDualUse
		case constraint[obj]:
			c.modes[obj] = UnionConstraintOnly
		default:
			c.modes[obj] = UnionValueOnly
		}
	}
	return c
}

func resolveUnionRef(obj types2.Object) *types2.TypeName {
	tn, ok := obj.(*types2.TypeName)
	if !ok || tn == nil {
		return nil
	}
	if tn.IsAlias() {
		if n, ok := types2.Unalias(tn.Type()).(*types2.Named); ok && n != nil {
			return n.Obj()
		}
		return nil
	}
	return tn
}

func bareRefName(e syntax.Expr) *syntax.Name {
	switch e := unwrapParen(e).(type) {
	case *syntax.Name:
		return e
	case *syntax.SelectorExpr:
		return e.Sel
	default:
		return nil
	}
}

func unionNameRef(info *types2.Info, e syntax.Expr) *types2.TypeName {
	if name := bareRefName(e); name != nil {
		return resolveUnionRef(info.Uses[name])
	}
	return nil
}

func NativeUnionEdits(info *types2.Info, files []*syntax.File, src map[*syntax.File][]byte) map[*syntax.File][]Edit {
	return unionEjectEdits(classifyUnions(info, files), src)
}

func EjectUnionEdits(info *types2.Info, files []*syntax.File, src map[*syntax.File][]byte) (map[*syntax.File][]Edit, error) {
	c := classifyUnions(info, files)
	var dual []string
	for obj, use := range c.modes {
		if use == UnionDualUse {
			dual = append(dual, obj.Name())
		}
	}
	if len(dual) > 0 {
		slices.Sort(dual)
		return nil, errors.New(DualUseMessage(dual[0]))
	}
	return unionEjectEdits(c, src), nil
}

func UnionImportEdits(info *types2.Info, src map[*syntax.File][]byte, lowered, union map[*syntax.File][]Edit) map[*syntax.File][]Edit {
	out := make(map[*syntax.File][]Edit)
	for file, erasures := range union {
		data, ok := src[file]
		if !ok {
			continue
		}
		if blanks := blankErasedImports(info, file, data, lowered[file], erasures); len(blanks) > 0 {
			out[file] = blanks
		}
	}
	return out
}

func blankErasedImports(info *types2.Info, file *syntax.File, data []byte, lowered, erasures []Edit) []Edit {
	starts := lineStarts(data)
	at := func(p syntax.Pos) int {
		line := int(p.Line())
		if !p.IsKnown() || line > len(starts) {
			return -1
		}
		return starts[line-1] + int(p.Col()) - 1
	}
	var erased [][2]int
	for _, e := range erasures {
		if s, t := at(e.Start), at(e.End); e.Text == "" && t > s {
			erased = append(erased, [2]int{s, t})
		}
	}
	used := make(map[types2.Object]bool)
	var visit func(syntax.Node) bool
	visit = func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.ImportDecl:
			return false
		case *syntax.SelectorExpr:
			syntax.Inspect(n.X, visit)
			return false
		case *syntax.Name:
			off := at(n.Pos())
			if !slices.ContainsFunc(erased, func(s [2]int) bool { return off >= s[0] && off < s[1] }) {
				used[info.Uses[n]] = true
			}
		}
		return true
	}
	syntax.Inspect(file, visit)
	var generated map[string]bool
	generatedUse := func(local string) bool {
		if !slices.ContainsFunc(lowered, func(e Edit) bool { return containsIdent(e.Text, local) }) {
			return false
		}
		if generated == nil {
			generated = make(map[string]bool)
			all := append(slices.Clone(lowered), erasures...)
			text, err := ApplyEdits(data, file.Pos().FileBase(), all)
			if err != nil {
				panic("logos: union import edits overlap lowering edits: " + err.Error())
			}
			f, err := parser.ParseFile(token.NewFileSet(), "", text, 0)
			if err != nil {
				panic("logos: lowered union import source is not Go syntax: " + err.Error())
			}
			if f != nil {
				ast.Inspect(f, func(n ast.Node) bool {
					if sel, ok := n.(*ast.SelectorExpr); ok {
						if id, ok := sel.X.(*ast.Ident); ok && id.Obj == nil {
							generated[id.Name] = true
						}
					}
					return true
				})
			}
		}
		return generated[local]
	}
	var blanks []splice
	for _, decl := range file.DeclList {
		imp, ok := decl.(*syntax.ImportDecl)
		if !ok || imp.Path == nil {
			continue
		}
		var obj types2.Object
		if imp.LocalPkgName != nil {
			obj = info.Defs[imp.LocalPkgName]
		} else {
			obj = info.Implicits[imp]
		}
		pn, ok := obj.(*types2.PkgName)
		if !ok || pn.Name() == "_" {
			continue
		}
		local := pn.Name()
		var kept bool
		if local == "." {
			for obj := range used {
				if obj != nil && obj.Pkg() == pn.Imported() && obj.Parent() == pn.Imported().Scope() {
					kept = true
					break
				}
			}
		} else {
			kept = used[pn] || generatedUse(local)
		}
		if kept {
			continue
		}
		start := at(imp.Path.Pos())
		if imp.LocalPkgName != nil {
			start = at(imp.LocalPkgName.Pos())
		}
		end := at(imp.Path.Pos()) + len(imp.Path.Value)
		if slices.ContainsFunc(lowered, func(e Edit) bool { return at(e.Start) <= end && at(e.End) >= start }) {
			continue
		}
		if imp.LocalPkgName != nil {
			blanks = append(blanks, splice{start: start, end: start + len(imp.LocalPkgName.Value), text: "_"})
		} else {
			blanks = append(blanks, splice{start: start, end: start, text: "_ "})
		}
	}
	return editsOf(data, file.Pos().FileBase(), blanks)
}

func unionEjectEdits(c *unionClass, src map[*syntax.File][]byte) map[*syntax.File][]Edit {
	out := make(map[*syntax.File][]Edit)
	for obj, use := range c.modes {
		if use != UnionValueOnly {
			continue
		}
		ud, ok := c.decls[obj]
		if !ok {
			continue
		}
		data, ok := src[ud.file]
		if !ok {
			continue
		}
		under, ok := obj.Type().Underlying().(*types2.Interface)
		if !ok || !isClosedUnion(under) {
			continue
		}
		splices, ok := unionSplices(ud, data)
		if !ok {
			continue
		}
		base := ud.file.Pos().FileBase()
		out[ud.file] = append(out[ud.file], editsOf(data, base, splices)...)
	}
	return out
}

func unionSplices(ud *unionDecl, data []byte) ([]splice, bool) {
	starts := lineStarts(data)
	off := func(p syntax.Pos) int {
		return starts[p.Line()-1] + int(p.Col()) - 1
	}
	closeOff := interfaceClose(data, off(ud.iface.Pos()))
	var splices []splice
	var terms []string
	for i, f := range ud.iface.MethodList {
		if f == nil || f.Type == nil {
			continue
		}
		op, ok := unwrapParen(f.Type).(*syntax.Operation)
		if !ok || op.Op != syntax.Or {
			continue
		}
		start := off(unionStart(f.Type))
		limit := closeOff
		if i+1 < len(ud.iface.MethodList) {
			if next := ud.iface.MethodList[i+1]; next != nil && next.Type != nil {
				limit = off(fieldPos(next))
			}
		}
		end := lastTokenEnd(data, start, limit)
		if end <= start {
			return nil, false
		}
		terms = append(terms, string(data[start:end]))
		splices = append(splices, splice{start: start, end: end})
	}
	if len(terms) == 0 {
		return nil, false
	}
	end := stmtEnd(data, off(ud.decl.Pos()))
	end = extendPastLineComment(data, end)
	comment := "\n// loGo/s union: " + strings.Join(terms, "; ")
	return append(splices, splice{start: end, end: end, text: comment}), true
}

func fieldPos(f *syntax.Field) syntax.Pos {
	if f.Name != nil {
		return f.Name.Pos()
	}
	return unionStart(f.Type)
}

func unionStart(e syntax.Expr) syntax.Pos {
	for {
		switch n := e.(type) {
		case *syntax.Operation:
			if n.Op != syntax.Or || n.X == nil {
				return e.Pos()
			}
			e = n.X
		case *syntax.SelectorExpr:
			if n.X == nil {
				return e.Pos()
			}
			e = n.X
		case *syntax.IndexExpr:
			if n.X == nil {
				return e.Pos()
			}
			e = n.X
		default:
			return e.Pos()
		}
	}
}

func interfaceClose(src []byte, from int) int {
	end := len(src)
	depth := 0
	seen := false
	scanSegment(src, from, len(src), func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		switch t.kind {
		case token.LBRACE:
			depth++
			seen = true
		case token.RBRACE:
			depth--
			if seen && depth == 0 {
				end = t.start
				return false
			}
		}
		return true
	})
	return end
}

func extendPastLineComment(src []byte, end int) int {
	nl := bytes.IndexByte(src[end:], '\n')
	lineEnd := len(src)
	if nl >= 0 {
		lineEnd = end + nl
	}
	seg := src[end:lineEnd]
	head, _, found := bytes.Cut(seg, []byte("//"))
	if !found {
		return end
	}
	if bytes.Contains(head, []byte("\"")) || bytes.Contains(head, []byte("`")) || bytes.Contains(head, []byte("/*")) {
		return end
	}
	return lineEnd
}
