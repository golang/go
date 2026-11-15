// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

type pendingSite struct {
	kind  string
	start int
	end   int
	fixed *int
}

type hintCollector struct {
	info *types2.Info
	file *hintFile
	qual types2.Qualifier
}

func (h *Hints) AddFile(pkg *types2.Package, info *types2.Info, file *syntax.File, src []byte) {
	f := newHintFile(file.Pos().FileBase().Filename(), src)
	h.byFile[file] = f
	h.byName[f.hints.File] = f
	h.byName[hintFileKey(f.hints.File)] = f
	c := &hintCollector{info: info, file: f, qual: hintQualifier(pkg)}
	syntax.Inspect(file, c.visit)
}

func hintQualifier(pkg *types2.Package) types2.Qualifier {
	return func(p *types2.Package) string {
		if p == pkg {
			return ""
		}
		return p.Name()
	}
}

func (c *hintCollector) visit(n syntax.Node) bool {
	switch n := n.(type) {
	case *syntax.Name:
		c.narrowedUse(n)
	case *syntax.SwitchStmt:
		c.unionSwitch(n.Tag)
	case *syntax.SwitchExpr:
		c.unionSwitch(n.Tag)
	case *syntax.InterpExpr:
		c.interp(n)
	case *syntax.TryExpr:
		c.try(n)
	}
	return true
}

func (f *hintFile) offset(pos syntax.Pos) (int, bool) {
	return f.index.offsetOf(int(pos.Line()), int(pos.Col()))
}

func (f *hintFile) rangeOf(start, end int) HintRange {
	return HintRange{Start: f.index.pos(start), End: f.index.pos(end)}
}

func (c *hintCollector) narrowedUse(n *syntax.Name) {
	v, ok := c.info.Uses[n].(*types2.Var)
	if !ok || hintOption(v.Type()) == nil {
		return
	}
	used := recordedType(c.info, n)
	if used == nil || types2.Identical(used, v.Type()) {
		return
	}
	off, ok := c.file.offset(n.Pos())
	if !ok {
		return
	}
	c.file.hints.Narrowed = append(c.file.hints.Narrowed, NarrowHint{
		Position: c.file.index.pos(off + len(n.Value)),
		Name:     n.Value,
		Type:     types2.TypeString(used, c.qual),
	})
}

func hintOption(t types2.Type) *types2.Option {
	switch t := types2.Unalias(t).(type) {
	case *types2.Option:
		return t
	case *types2.Named:
		o, _ := t.Underlying().(*types2.Option)
		return o
	}
	return nil
}

func recordedType(info *types2.Info, e syntax.Expr) types2.Type {
	if info.Types != nil {
		return info.Types[e].Type
	}
	return e.GetTypeInfo().Type
}

func (c *hintCollector) unionSwitch(tag syntax.Expr) {
	guard, ok := tag.(*syntax.TypeSwitchGuard)
	if !ok {
		return
	}
	t := c.info.TypeOf(guard.X)
	members := unionMembers(t)
	off, ok := c.file.offset(guard.Pos())
	if members == nil || !ok {
		return
	}
	names := make([]string, len(members))
	for i, m := range members {
		names[i] = types2.TypeString(m, c.qual)
	}
	c.file.hints.Unions = append(c.file.hints.Unions, UnionHint{
		Position: c.file.index.pos(off),
		Type:     types2.TypeString(t, c.qual),
		Members:  names,
	})
}

func unionMembers(t types2.Type) []types2.Type {
	if t == nil {
		return nil
	}
	iface, ok := types2.Unalias(t).Underlying().(*types2.Interface)
	if !ok || !isClosedUnion(iface) {
		return nil
	}
	var members []types2.Type
	for i := range iface.NumEmbeddeds() {
		u, ok := iface.EmbeddedType(i).(*types2.Union)
		if !ok {
			continue
		}
		for j := range u.Len() {
			members = append(members, u.Term(j).Type())
		}
	}
	return members
}

func (c *hintCollector) interp(e *syntax.InterpExpr) {
	kinds := make([]types2.LogosInterpKind, len(e.Exprs))
	for i, op := range e.Exprs {
		k, ok := types2.LogosInterpKindOf(c.info.TypeOf(op))
		if !ok {
			return
		}
		kinds[i] = k
	}
	start, ok := c.file.offset(e.Pos())
	if !ok {
		return
	}
	fixed := InterpFixedBytes(e.Parts, kinds)
	c.file.pending = append(c.file.pending, pendingSite{kind: SiteInterp, start: start, end: start + len(e.Raw), fixed: &fixed})
}

func (c *hintCollector) try(t *syntax.TryExpr) {
	start, ok := c.file.offset(t.Pos())
	if !ok {
		return
	}
	c.file.pending = append(c.file.pending, pendingSite{kind: SiteTry, start: start, end: start + 1})
}

type editSpan struct {
	start int
	end   int
	text  string
}

func (f *hintFile) editSpans(edits []Edit) []editSpan {
	var spans []editSpan
	for _, e := range edits {
		start, ok1 := f.offset(e.Start)
		end, ok2 := f.offset(e.End)
		if ok1 && ok2 && end > start {
			spans = append(spans, editSpan{start: start, end: end, text: e.Text})
		}
	}
	return spans
}

func coveringEdit(spans []editSpan, off int) (editSpan, bool) {
	for _, s := range spans {
		if s.start <= off && off < s.end {
			return s, true
		}
	}
	return editSpan{}, false
}

func (h *Hints) Lowered(file *syntax.File, edits []Edit) {
	f := h.byFile[file]
	if f == nil {
		return
	}
	spans := f.editSpans(edits)
	for _, p := range f.pending {
		if e, ok := coveringEdit(spans, p.start); ok {
			f.hints.Sites = append(f.hints.Sites, SiteHint{Range: f.rangeOf(p.start, p.end), Kind: p.kind, Lowered: e.text, FixedBytes: p.fixed})
		}
	}
	f.pending = nil
	for _, e := range spans {
		f.constSite(e)
	}
}

func (f *hintFile) constSite(e editSpan) {
	kind, end, ok := constKeywords(f.index.src, e.start)
	if !ok {
		return
	}
	f.hints.Sites = append(f.hints.Sites, SiteHint{Range: f.rangeOf(e.start, end), Kind: kind, Lowered: e.text})
}

func constKeywords(src []byte, off int) (kind string, end int, ok bool) {
	rest := src[off:]
	if !bytes.HasPrefix(rest, []byte("const")) {
		return "", 0, false
	}
	word := len("const")
	for word < len(rest) && isSpace(rest[word]) {
		word++
	}
	if word == len("const") {
		return "", 0, false
	}
	stop := word
	for stop < len(rest) && rest[stop] >= 'a' && rest[stop] <= 'z' {
		stop++
	}
	switch string(rest[word:stop]) {
	case "for":
		return SiteConstFor, off + stop, true
	case "switch":
		return SiteConstSwitch, off + stop, true
	}
	return "", 0, false
}

func (h *Hints) Rodata(file *syntax.File) {
	f := h.byFile[file]
	if f == nil {
		return
	}
	for _, decl := range file.DeclList {
		vd, ok := decl.(*syntax.VarDecl)
		if !ok {
			continue
		}
		for _, name := range vd.NameList {
			if IsRodata(h.pkgPath, name.Value) {
				f.rodata(name)
			}
		}
	}
}

func (f *hintFile) rodata(name *syntax.Name) {
	off, ok := f.offset(name.Pos())
	if !ok {
		return
	}
	f.addAlloc(AllocHint{
		Position: f.index.pos(off),
		Tier:     TierRodata,
		Message:  name.Value + " is placed in read-only data",
	})
}
