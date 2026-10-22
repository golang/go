// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"slices"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"internal/buildcfg"
)

// concurrentMultiwordRace is concurrency::concurrent_multiword_race (SPEC
// Sec 1.2, #13, #148): a local variable of multi-word type that a child
// captures and either side writes after the spawn. A child is the function
// literal of a go statement or a loGo/s group child block. A variable is
// multi-word when its value spans more than one machine word: a string
// (two), a slice (three), an interface or union (two), a complex number
// (two), and a struct or array whose value-bearing storage, padding and
// zero-sized fields excluded, spans more than one target machine word. A value
// of a logos/sync type is exempt, and so is a const view, which nothing
// writes through. A write is an assignment, an increment, or a range
// assignment to the variable itself or to a field or array element held in
// it; storing through a pointer, a slice element, or a map entry writes other
// storage. The spawn starts the write window, which ends with the
// innermost enclosing go group block, because the group joins its children
// there, and otherwise with the enclosing declaration. A write by the child
// is inside the window. The rule reads syntax and checker objects only, so
// it holds for Go packages through the go statement leg and for loGo/s
// packages through both legs.
var concurrentMultiwordRace = Rule{
	ID:    "concurrency::concurrent_multiword_race",
	Group: Concurrency,
	Go:    true,
	Check: checkConcurrentMultiwordRace,
}

// spawn is one child of a declaration: its statement and the node whose
// span is the child's own scope.
type spawn struct {
	node  syntax.Node
	child syntax.Node
}

// varWrite is one assignment to a variable.
type varWrite struct {
	v   *types2.Var
	pos syntax.Pos
}

// raceCheck is the analysis of one top-level declaration.
type raceCheck struct {
	pass   *Pass
	end    syntax.Pos
	writes []varWrite
	groups []*syntax.GroupStmt
	spawns []spawn
}

// checkConcurrentMultiwordRace is the Check of concurrentMultiwordRace.
func checkConcurrentMultiwordRace(pass *Pass) {
	for _, f := range pass.Files {
		for _, d := range f.DeclList {
			c := &raceCheck{pass: pass, end: syntax.EndPos(d)}
			c.collect(d)
			c.report()
		}
	}
}

// collect records the writes, groups, and children under root.
func (c *raceCheck) collect(root syntax.Node) {
	syntax.Inspect(root, func(n syntax.Node) bool {
		c.noteWrites(n)
		switch n := n.(type) {
		case *syntax.GroupStmt:
			c.groups = append(c.groups, n)
		case *syntax.GroupChild:
			c.spawns = append(c.spawns, spawn{node: n, child: n})
		case *syntax.CallStmt:
			if lit := goLiteral(n); lit != nil {
				c.spawns = append(c.spawns, spawn{node: n, child: lit})
			}
		}
		return true
	})
}

// goLiteral returns the function literal that go statement s calls, or nil.
func goLiteral(s *syntax.CallStmt) *syntax.FuncLit {
	call := callOf(s.Call)
	if s.Tok != syntax.Go || call == nil {
		return nil
	}
	fun := call.Fun
	for {
		p, ok := fun.(*syntax.ParenExpr)
		if !ok {
			break
		}
		fun = p.X
	}
	lit, _ := fun.(*syntax.FuncLit)
	return lit
}

// noteWrites records the variables that statement n assigns.
func (c *raceCheck) noteWrites(n syntax.Node) {
	var targets []syntax.Expr
	switch n := n.(type) {
	case *syntax.AssignStmt:
		targets = exprList(n.Lhs)
	case *syntax.RangeClause:
		if n.Lhs != nil && !n.Def {
			targets = exprList(n.Lhs)
		}
	}
	for _, t := range targets {
		if v := c.writtenVar(t); v != nil {
			c.writes = append(c.writes, varWrite{v: v, pos: syntax.StartPos(n)})
		}
	}
}

// writtenVar returns the variable whose own storage an assignment to e
// stores into, or nil.
func (c *raceCheck) writtenVar(e syntax.Expr) *types2.Var {
	for e != nil {
		if name, ok := e.(*syntax.Name); ok {
			v, _ := c.pass.Info.Uses[name].(*types2.Var)
			return v
		}
		e = c.inPlaceOperand(e)
	}
	return nil
}

// inPlaceOperand returns the operand x of a parenthesized expression, a
// field selection, or an array index e when e names storage inside x itself,
// and nil otherwise.
func (c *raceCheck) inPlaceOperand(e syntax.Expr) syntax.Expr {
	switch x := e.(type) {
	case *syntax.ParenExpr:
		return x.X
	case *syntax.SelectorExpr:
		sel := c.pass.Info.Selections[x]
		if sel != nil && sel.Kind() == types2.FieldVal && !sel.Indirect() {
			return x.X
		}
	case *syntax.IndexExpr:
		if c.isArray(TypeOf(x.X)) {
			return x.X
		}
	}
	return nil
}

// isArray reports whether t is an array type.
func (c *raceCheck) isArray(t types2.Type) bool {
	if t == nil {
		return false
	}
	_, ok := t.Underlying().(*types2.Array)
	return ok
}

// report flags every captured multi-word variable that a child shares with
// a write.
func (c *raceCheck) report() {
	for _, s := range c.spawns {
		for _, v := range c.captured(s.child) {
			if multiWord(v.Type()) && c.written(s, v) {
				c.flag(s, v)
			}
		}
	}
}

// captured returns the local variables the child references that are
// declared outside it, in order of first reference.
func (c *raceCheck) captured(child syntax.Node) []*types2.Var {
	var out []*types2.Var
	start, end := syntax.StartPos(child), syntax.EndPos(child)
	syntax.Inspect(child, func(n syntax.Node) bool {
		name, ok := n.(*syntax.Name)
		if !ok {
			return true
		}
		v, _ := c.pass.Info.Uses[name].(*types2.Var)
		if v != nil && c.isLocal(v) && !within(v.Pos(), start, end) && !slices.Contains(out, v) {
			out = append(out, v)
		}
		return true
	})
	return out
}

// isLocal reports whether v is a variable of a function, not a field or a
// package-level variable.
func (c *raceCheck) isLocal(v *types2.Var) bool {
	scope := v.Parent()
	return !v.IsField() && scope != nil && scope != c.pass.Pkg.Scope()
}

// within reports whether p lies between start and end, inclusive.
func within(p, start, end syntax.Pos) bool {
	return p.IsKnown() && p.Cmp(start) >= 0 && p.Cmp(end) <= 0
}

// written reports whether v is written inside the window of s.
func (c *raceCheck) written(s spawn, v *types2.Var) bool {
	start, end := syntax.StartPos(s.node), c.window(s)
	return slices.ContainsFunc(c.writes, func(w varWrite) bool {
		return w.v == v && within(w.pos, start, end)
	})
}

// window returns the end of the window in which a write races with the
// child of s: the end of the innermost group that holds s, or else the end
// of the declaration.
func (c *raceCheck) window(s spawn) syntax.Pos {
	end := c.end
	start := syntax.StartPos(s.node)
	for _, g := range c.groups {
		gend := syntax.EndPos(g)
		if within(start, syntax.StartPos(g), gend) && gend.Cmp(end) < 0 {
			end = gend
		}
	}
	return end
}

// flag reports that the child of s shares v with a write.
func (c *raceCheck) flag(s spawn, v *types2.Var) {
	typ := types2.TypeString(v.Type(), types2.RelativeTo(c.pass.Pkg))
	c.pass.Report(Diagnostic{
		Pos:     syntax.StartPos(s.node),
		End:     syntax.EndPos(s.node),
		Message: "variable " + v.Name() + " of multi-word type " + typ + " is shared with this child and written; guard it with logos/sync or pass a const view",
		Provenance: Provenance{
			OriginExpr:   v.Name(),
			OriginLine:   int(v.Pos().Line()),
			InferredType: typ,
		},
	})
}

// multiWord reports whether a variable of type t races when a child shares
// it with a write: its value spans more than one word and t is not a
// logos/sync type.
func multiWord(t types2.Type) bool {
	return !isLogosSyncType(t) && wordsOf(t) > 1
}

// isLogosSyncType reports whether t is a type declared by logos/sync.
func isLogosSyncType(t types2.Type) bool {
	named, ok := types2.Unalias(t).(*types2.Named)
	if !ok {
		return false
	}
	pkg := named.Obj().Pkg()
	return pkg != nil && pkg.Path() == logosSyncPath
}

// wordsOf returns the machine words a value of type t spans: its
// value-bearing bytes over the target word size, rounded up once.
func wordsOf(t types2.Type) int {
	word := targetWord()
	return int((valueBytes(t, word) + word - 1) / word)
}

// targetWord returns the target machine word size in bytes: the size the
// gc layout gives an int for the configured GOARCH, which is one word on
// every architecture, or eight when the layout is unknown.
func targetWord() int64 {
	sizes := types2.SizesFor("gc", buildcfg.GOARCH)
	if sizes == nil {
		return 8
	}
	if word := sizes.Sizeof(types2.Typ[types2.Int]); word > 0 {
		return word
	}
	return 8
}

// valueBytes returns the value-bearing bytes a value of type t holds:
// padding and zero-sized fields count nothing, so a struct or array sums
// the true sizes of its parts. A pointer, a map, a channel, a function, a
// const view, and a type parameter count one word.
func valueBytes(t types2.Type, word int64) int64 {
	if _, ok := types2.Unalias(t).(*types2.TypeParam); ok {
		return word
	}
	switch u := t.Underlying().(type) {
	case *types2.Basic:
		return basicBytes(u, word)
	case *types2.Slice:
		return 3 * word
	case *types2.Interface:
		return 2 * word
	case *types2.Array:
		return min(u.Len()*valueBytes(u.Elem(), word), 2*word)
	case *types2.Struct:
		total := int64(0)
		for i := range u.NumFields() {
			total += valueBytes(u.Field(i).Type(), word)
		}
		return total
	}
	return word
}

// basicBytes returns the value-bearing bytes of a basic type: a string
// takes its two-word header, a complex number its exact width, a narrow
// scalar its true size, and int, uint, and uintptr one word.
func basicBytes(b *types2.Basic, word int64) int64 {
	if b.Info()&types2.IsString != 0 {
		return 2 * word
	}
	switch b.Kind() {
	case types2.Complex128:
		return 16
	case types2.Complex64:
		return 8
	case types2.Bool, types2.Int8, types2.Uint8:
		return 1
	case types2.Int16, types2.Uint16:
		return 2
	case types2.Int32, types2.Uint32, types2.Float32:
		return 4
	case types2.Int64, types2.Uint64, types2.Float64:
		return 8
	}
	return word
}
