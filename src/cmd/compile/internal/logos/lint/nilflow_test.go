// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"slices"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
)

// flowSrc holds return statements over the path p. A line marked "// proved"
// is a return where the walk must hold p, or the path the line returns, as
// non-nil; every other return must not.
const flowSrc = `package p

import "os"

type T struct{ P, P2 *int }

func ifNotNil(p *int) *int {
	if p != nil {
		return p // proved
	}
	return p
}

func ifNil(p *int) *int {
	if p == nil {
		return p
	}
	return p // proved
}

func ifNilNoExit(p *int) *int {
	if p == nil {
		_ = p
	}
	return p
}

func elseOfNil(p *int) *int {
	if p == nil {
		return nil
	} else {
		return p // proved
	}
}

func elseOfNotNil(p *int) *int {
	if p != nil {
		return nil
	} else {
		return p
	}
}

func elseIfChain(p *int, ok bool) *int {
	if p == nil {
		return nil
	} else if ok {
		return p // proved
	}
	return p // proved
}

func andChain(p *int, ok bool) *int {
	if p != nil && ok {
		return p // proved
	}
	return p
}

func orChain(p *int, ok bool) *int {
	if p != nil || ok {
		return p
	}
	return p
}

func orExit(p *int, ok bool) *int {
	if p == nil || ok {
		return nil
	}
	return p // proved
}

func andExit(p *int, ok bool) *int {
	if p == nil && ok {
		return nil
	}
	return p
}

func negated(p *int) *int {
	if !(p == nil) {
		return p // proved
	}
	return p
}

func negatedExit(p *int) *int {
	if !(p != nil) {
		return nil
	}
	return p // proved
}

func nilOnLeft(p *int) *int {
	if nil != p {
		return p // proved
	}
	return p
}

func twoPaths(p, q *int) *int {
	if q != nil {
		return p
	}
	return p
}

func assignKills(p, q *int) *int {
	if p == nil {
		return nil
	}
	p = q
	return p
}

func tupleAssignKills(p *int, f func() (*int, error)) *int {
	if p == nil {
		return nil
	}
	p, _ = f()
	return p
}

func incrementedElementKeeps(p *int, xs []int) *int {
	if p == nil {
		return nil
	}
	xs[0]++
	return p // proved
}

func storeThroughPointerKeeps(p *int) *int {
	if p == nil {
		return nil
	}
	*p = 1
	return p // proved
}

func assignInBranchKills(p, q *int, ok bool) *int {
	if p == nil {
		return nil
	}
	if ok {
		p = q
	}
	return p
}

func assignInClosureKills(p *int) *int {
	if p == nil {
		return nil
	}
	clear := func() { p = nil }
	clear()
	return p
}

func shadowKills(p, q *int) *int {
	if p == nil {
		return nil
	}
	{
		p := q
		return p
	}
}

func declKills(p *int) *int {
	if p == nil {
		return nil
	}
	{
		var p *int
		return p
	}
}

func reassignThenCheck(p, q *int) *int {
	p = q
	if p != nil {
		return p // proved
	}
	return p
}

func loopAssignKills(p *int, next []*int) *int {
	if p == nil {
		return nil
	}
	for _, n := range next {
		p = n
	}
	return p
}

func loopConditionProves(p *int) *int {
	for p != nil {
		return p // proved
	}
	return p
}

func loopKeepsUnwrittenProof(p *int, ok bool) *int {
	if p == nil {
		return nil
	}
	for ok {
		return p // proved
	}
	return p // proved
}

func loopWritesInBody(p, q *int, ok bool) *int {
	if p == nil {
		return nil
	}
	for ok {
		return p
		p = q
	}
	return p
}

func panicExit(p *int) *int {
	if p == nil {
		panic("nil")
	}
	return p // proved
}

func processExit(p *int) *int {
	if p == nil {
		os.Exit(1)
	}
	return p // proved
}

func breakExit(p *int, n int) *int {
	for i := 0; i < n; i++ {
		if p == nil {
			break
		}
		return p // proved
	}
	return p
}

func continueExit(p *int, n int) *int {
	for i := 0; i < n; i++ {
		if p == nil {
			continue
		}
		return p // proved
	}
	return nil
}

func bothBranchesExit(p *int, ok bool) *int {
	if p == nil {
		return nil
	} else if ok {
		return p // proved
	} else {
		return p // proved
	}
}

func nestedBlockExit(p *int) *int {
	if p == nil {
		{
			return nil
		}
	}
	return p // proved
}

func initStatement(p *int, f func() *int) *int {
	if q := f(); q != nil {
		return p
	}
	return p
}

func initShadows(p *int, f func() *int) *int {
	if p == nil {
		return nil
	}
	if p := f(); true {
		return p
	}
	return p
}

func switchKeepsProof(p *int, n int) *int {
	if p == nil {
		return nil
	}
	switch n {
	case 1:
		return p // proved
	default:
		return p // proved
	}
}

func switchWriteKills(p, q *int, n int) *int {
	if p == nil {
		return nil
	}
	switch n {
	case 1:
		p = q
	}
	return p
}

func selectKeepsProof(p *int, c chan int) *int {
	if p == nil {
		return nil
	}
	select {
	case <-c:
		return p // proved
	}
}

func labeled(p *int) *int {
	if p == nil {
		return nil
	}
L:
	for {
		return p // proved
		break L
	}
	return p // proved
}

func fieldPath(h T) *int {
	if h.P != nil {
		return h.P // proved
	}
	return h.P
}

func fieldPathKilledByBase(h, g T) *int {
	if h.P == nil {
		return nil
	}
	h = g
	return h.P
}

func fieldPathKilledByField(h T, q *int) *int {
	if h.P == nil {
		return nil
	}
	h.P = q
	return h.P
}

func siblingFieldKept(h T, q *int) *int {
	if h.P == nil {
		return nil
	}
	h.P2 = q
	return h.P // proved
}

func packagePath() *os.File {
	if os.Stdout == nil {
		return nil
	}
	return os.Stdout // proved
}

func callIsNeverProved(f func() *int) *int {
	if f() != nil {
		return f()
	}
	return f()
}

func closureBodyIsNotWalked(p *int) func() *int {
	if p == nil {
		return nil
	}
	return func() *int {
		return p
	}
}

func siblingBlockExit(p *int) *int {
	{
		if p == nil {
			return nil
		}
	}
	return p // proved
}

func siblingBlockKilled(p, q *int) *int {
	{
		if p == nil {
			return nil
		}
		p = q
	}
	return p
}
`

// TestNilFlow checks the facts the control-flow walk holds at each return:
// condition polarity, early exits, and every kind of write that ends a proof.
func TestNilFlow(t *testing.T) {
	in := check(t, KindGo, "flow.go", flowSrc)
	pass := &Pass{Info: in.Info, Pkg: in.Pkg, Files: in.Files, Kind: in.Kind, in: in}
	var got []int
	w := &flow{pass: pass, onReturn: func(s *syntax.ReturnStmt, facts nilFacts) {
		if key := pathKey(s.Results); key != "" && facts[key] {
			got = append(got, int(syntax.StartPos(s).Line()))
		}
	}}
	for _, d := range in.Files[0].DeclList {
		if fd, ok := d.(*syntax.FuncDecl); ok {
			w.stmts(fd.Body.List, nilFacts{})
		}
	}
	var want []int
	for i, line := range strings.Split(flowSrc, "\n") {
		if strings.Contains(line, "// proved") {
			want = append(want, i+1)
		}
	}
	if !slices.Equal(got, want) {
		t.Errorf("proved returns:\n got %v\nwant %v", got, want)
	}
}
