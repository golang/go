// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"errors"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// Eject writes one ejected Go file per logos source into dir by applying
// edits to the original bytes. Files without a src entry are skipped.
// Every eject output comes from the same lowered nodes as the native
// build, so comments outside replacement spans survive verbatim. Beside
// each ejected file it writes the edit map of that file (see EditMap), so
// a tool can translate positions between the original and ejected texts.
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
		out, spans, err := ApplyEditsMapped(data, base, edits[file])
		if err != nil {
			return err
		}
		name := filepath.Join(dir, ejectedName(base.Filename()))
		if err := os.WriteFile(name, out, 0o666); err != nil {
			return err
		}
		if err := writeEditMap(name+EditMapSuffix, base.Filename(), len(data), len(out), spans); err != nil {
			return err
		}
	}
	return nil
}

// ejectedName maps a logos file name to its ejected Go file name.
func ejectedName(name string) string {
	base := filepath.Base(name)
	return strings.TrimSuffix(base, filepath.Ext(base)) + ".go"
}

// DualUseMessage is the unprefixed diagnostic for a declared union that
// is used both as a type constraint and as a value type. Such a union
// has no Go form: a constraint needs the union terms, a value needs the
// methods-only interface. Callers add the command prefix exactly once
// ("logos eject: %v" in writeLogosEject, "logos build: %s" at a Go
// import), so the message itself carries no prefix.
func DualUseMessage(name string) string {
	return "union " + name + " is used as a type constraint and as a value type; Go cannot express both"
}

// UnionUse describes how one declared union is used within its package.
type UnionUse int

const (
	// UnionConstraintOnly is named only in constraint positions (type
	// parameter bounds, including unions embedded in generic
	// interfaces). It keeps its union form in eject text and stays out
	// of __.PKGDEF, as today.
	UnionConstraintOnly UnionUse = iota
	// UnionValueOnly is never named in a constraint position. This
	// covers declared-but-unreferenced unions too: with no constraint
	// use the methods-only form is valid Go. It ejects and exports
	// methods-only.
	UnionValueOnly
	// UnionDualUse is named in both positions and has no Go form. Eject
	// fails, and a Go importer fails at the import. Native builds keep
	// the full union.
	UnionDualUse
)

// KeepUnionDecl reports whether the export-data writer must emit a
// declaration for obj. Value-used unions (dual included) need a Go form
// in both roots; constraint-only unions keep today's skip.
func KeepUnionDecl(m map[*types2.TypeName]UnionUse, obj *types2.TypeName) bool {
	u, ok := m[obj]
	return ok && u != UnionConstraintOnly
}

// HasUnionTerms reports whether iface directly embeds a union term list,
// open or closed. It mirrors the declared-form check the checker folds
// into type sets, using only the exported types2 API. Transitive terms
// (an interface embedding a named union) do not count: the embedded
// name carries the usage, and the eject and export rewrites of that
// name keep the outer declaration valid.
func HasUnionTerms(iface *types2.Interface) bool {
	for i := range iface.NumEmbeddeds() {
		if IsUnionTerm(iface.EmbeddedType(i)) {
			return true
		}
	}
	return false
}

// IsUnionTerm reports whether t is a union term list as embedded in an
// interface. The methods-only export drops exactly these; every other
// embedded type is kept.
func IsUnionTerm(t types2.Type) bool {
	u, ok := t.(*types2.Union)
	return ok && u.Len() > 0
}

// isClosedUnion reports whether iface is a closed union: direct terms
// exist and none of them is a ~ term.
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

// unionDecl records one rewritable union declaration: a package-level or
// local `type U interface{ ... terms ... }` with no type parameters.
type unionDecl struct {
	decl  *syntax.TypeDecl
	iface *syntax.InterfaceType
	file  *syntax.File
}

// unionClass is the full classification. ClassifyDeclaredUnions exposes
// only the modes; the eject builders use the declarations too.
type unionClass struct {
	modes map[*types2.TypeName]UnionUse
	decls map[*types2.TypeName]*unionDecl
}

// ClassifyDeclaredUnions returns, for each union declared in files, how
// the package uses it. It runs on the info of any successful check; the
// writer calls it with the final check, the eject hook with the first.
func ClassifyDeclaredUnions(info *types2.Info, files []*syntax.File) map[*types2.TypeName]UnionUse {
	return classifyUnions(info, files).modes
}

// classifyUnions collects declared unions and their constraint, value,
// and embedding references, then closes both usages along embedding
// edges. A use inside a type parameter list is a constraint use; a use
// embedded in another interface follows that interface (edge V -> U, so
// V's own usage decides U's); every other reference is a value use.
// References through an alias count for the aliased union. A `type Y V`
// declaration is not a union of its own: it lends Y's usage to V.
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
			if td.Alias || len(td.TParamList) != 0 {
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
				// A declaration is not a use: Y's own usage reaches
				// V through the edge, so the RHS name stays out of
				// the value walk. Otherwise a constraint-only Y
				// would force a spurious dual on V.
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
	// Usage is tracked per occurrence for every named type: edge
	// sources need their own status for the fixpoint, while only
	// unions get modes. A name in a bound counts as constraint use even
	// when another occurrence of the same object is a value use (that
	// is precisely dual use), so the bound and embedding passes mark
	// nodes, and the value walk below skips exactly those nodes.
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
	// Constraint positions: type parameter bounds of every declaration.
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
	// Embedding positions: unions named under an embedded element of a
	// non-generic named interface become edges; the same under a generic
	// interface counts as constraint use (a generic interface can only
	// constrain, up to instantiation subtleties eject cannot spell, so
	// the loud dual-use failure beats a silent semantic change).
	// walkEmbeddedNames calls mark for each name under an embedded
	// interface element, descending through parentheses and | chains.
	var walkEmbeddedNames func(e syntax.Expr, mark func(*syntax.Name))
	walkEmbeddedNames = func(e syntax.Expr, mark func(*syntax.Name)) {
		e = unwrapParens(e)
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

	// Value positions: every reference outside a bound, an embedding,
	// or a bare declaration RHS. A union referenced here and in a
	// bound is dual-use.
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

	// Close both usages along edges. Sources carry their own status,
	// so a union embedded in a value-used interface is value-used and
	// one embedded in a constraint is constraint-used.
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
			// Value-only or unreferenced: with no constraint use the
			// methods-only form is valid Go.
			c.modes[obj] = UnionValueOnly
		}
	}
	return c
}

// resolveUnionRef follows an object through aliases to the declared
// union it denotes, reporting nil for everything else. References
// through `type Y = U` count as references to U.
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

// bareRefName returns the name a `type Y V` declaration refers to, if
// any. SelectorExpr carries the qualifier separately; only Sel denotes
// the target.
func bareRefName(e syntax.Expr) *syntax.Name {
	switch e := unwrapParens(e).(type) {
	case *syntax.Name:
		return e
	case *syntax.SelectorExpr:
		return e.Sel
	default:
		return nil
	}
}

// unionNameRef resolves a bare type expression to the union it names,
// for `type Y V` declarations that lend their usage to V.
func unionNameRef(info *types2.Info, e syntax.Expr) *types2.TypeName {
	if name := bareRefName(e); name != nil {
		return resolveUnionRef(info.Uses[name])
	}
	return nil
}

// unwrapParens strips parenthesized wrappers the checker folds away.
func unwrapParens(e syntax.Expr) syntax.Expr {
	for {
		p, ok := e.(*syntax.ParenExpr)
		if !ok || p == nil {
			return e
		}
		e = p.X
	}
}

// NativeUnionEdits records the union eject edits for an ordinary build.
// A dual-use union is valid native code, so it never fails; the edits
// are kept for a later eject.
func NativeUnionEdits(info *types2.Info, files []*syntax.File, src map[*syntax.File][]byte) map[*syntax.File][]Edit {
	return unionEjectEdits(classifyUnions(info, files), src)
}

// EjectUnionEdits records the union eject edits for an eject. A
// dual-use union has no Go form, so it fails with DualUseMessage.
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

// unionEjectEdits builds the eject edits for a classification shared by
// NativeUnionEdits and EjectUnionEdits.
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
			// Open unions cannot occupy value positions in checked
			// code, so this is unreachable; leave the form alone.
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

// unionSplices builds the splices rewriting one union declaration: each
// union term group leaves the interface, and a comment naming the
// member set follows the declaration. Every splice replaces a token or
// inserts at a token boundary, so a sibling edit on the same offsets
// stays a loud applySplices error instead of a silent merge.
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
		op, ok := unwrapParens(f.Type).(*syntax.Operation)
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

// fieldPos returns the position opening an interface or struct field.
// A union term group opens at its leftmost leaf: both binaryExpr and
// embeddedElem stamp an Operation at its operator token, so the field
// position alone would drop the first member from the span and the
// comment.
func fieldPos(f *syntax.Field) syntax.Pos {
	if f.Name != nil {
		return f.Name.Pos()
	}
	return unionStart(f.Type)
}

// unionStart returns the position opening a union term group: the
// leftmost token of its | chain. An Operation sits at its operator, a
// SelectorExpr at its dot, and an IndexExpr at its bracket, so all
// three descend to X; every other node (including a parenthesized
// term, which opens at its parenthesis) reports its own position, so
// method fields keep working through fieldPos. Slice, assertion, and
// call nodes need no case: none is denotable in type position.
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

// interfaceClose returns the offset of the brace closing the interface
// literal that starts at from, by matching bracket depth over tokens.
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

// extendPastLineComment moves end past a // comment sharing its line,
// so the union comment lands on its own line instead of swallowing a
// trailing comment into itself. The trailing text survives verbatim
// either way; this keeps it readable. When in doubt it gives up and the
// insert stays at the declaration end.
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
