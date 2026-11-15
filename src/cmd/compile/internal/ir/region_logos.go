// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"internal/buildcfg"
	"strings"
	"sync"

	"cmd/compile/internal/types"
)

type LogosRegionOp uint8

const (
	LogosRegionNone LogosRegionOp = iota
	LogosRegionOpen
	LogosRegionNew
	LogosRegionMake
	LogosRegionAppend
)

const logosArenaPath = "logos/arena"

func LogosRegionsEnabled() bool {
	return buildcfg.Experiment.LogosArena
}

func logosRegionBaseName(sym *types.Sym) (string, bool) {
	if sym == nil || sym.Pkg == nil || sym.Pkg.Path != logosArenaPath {
		return "", false
	}
	name := sym.Name
	if i := strings.IndexByte(name, '['); i >= 0 {
		name = name[:i]
	}
	return name, true
}

func LogosRegionCallOp(call *CallExpr) LogosRegionOp {
	if call == nil || call.Fun == nil {
		return LogosRegionNone
	}
	callee := StaticCalleeName(call.Fun)
	if callee == nil {
		return LogosRegionNone
	}
	base, ok := logosRegionBaseName(callee.Sym())
	if !ok {
		return LogosRegionNone
	}
	switch base {
	case "Open":
		return LogosRegionOpen
	case "New":
		return LogosRegionNew
	case "Make":
		return LogosRegionMake
	case "Append":
		return LogosRegionAppend
	}
	return LogosRegionNone
}

func LogosRegionArgBase(call *CallExpr) int {
	callee := StaticCalleeName(call.Fun)
	if callee == nil {
		return 0
	}
	if strings.IndexByte(callee.Sym().Name, '[') >= 0 {
		return 1
	}
	return 0
}
func LogosRegionElem(call *CallExpr) *types.Type {
	if call == nil {
		return nil
	}
	t := call.Type()
	if t == nil {
		return nil
	}
	switch t.Kind() {
	case types.TPTR, types.TSLICE:
		return t.Elem()
	}
	return nil
}

var logosRegionNoInlineCache sync.Map

func LogosRegionNoInline(fn *Func) bool {
	if fn == nil {
		return false
	}
	if v, ok := logosRegionNoInlineCache.Load(fn); ok {
		return v.(bool)
	}
	res := logosRegionFuncNoInline(fn)
	v, _ := logosRegionNoInlineCache.LoadOrStore(fn, res)
	return v.(bool)
}

func logosRegionFuncNoInline(fn *Func) bool {
	if fn.Nname != nil {
		if name, ok := logosRegionBaseName(fn.Sym()); ok {
			switch name {
			case "Open", "New", "Make", "Append":
				return true
			}
		}
	}
	for _, st := range fn.Body {
		if logosDeepHasOpen(st) {
			return true
		}
	}
	return false
}

func logosDeepHasOpen(n Node) bool {
	return logosHasOpen(n, true)
}

func logosExprHasOpen(n Node) bool {
	return logosHasOpen(n, false)
}

func logosHasOpen(n Node, deep bool) bool {
	if n == nil {
		return false
	}
	if _, ok := n.(*ClosureExpr); ok {
		return false
	}
	if call, ok := n.(*CallExpr); ok && LogosRegionCallOp(call) == LogosRegionOpen {
		return true
	}
	if !deep {
		switch n.Op() {
		case OBLOCK, OFOR, ORANGE, OIF, OSWITCH, OSELECT:
			return false
		}
	}
	found := false
	DoChildren(n, func(c Node) bool {
		if logosHasOpen(c, deep) {
			found = true
			return true
		}
		return false
	})
	return found
}

type LogosRegionOwner struct {
	Stmt   Node
	List   *Nodes
	Parent *LogosRegionOwner
	Loop   bool
}

func (o *LogosRegionOwner) Encloses(p *LogosRegionOwner) bool {
	for ; p != nil; p = p.Parent {
		if p == o {
			return true
		}
	}
	return false
}

type logosRegionMaps struct {
	owners map[*CallExpr]*LogosRegionOwner
	decls  map[*Name]*LogosRegionOwner
}

var logosRegionMapsCache sync.Map

func logosRegionMapsOf(fn *Func) *logosRegionMaps {
	if v, ok := logosRegionMapsCache.Load(fn); ok {
		return v.(*logosRegionMaps)
	}
	maps := &logosRegionMaps{
		owners: make(map[*CallExpr]*LogosRegionOwner),
		decls:  make(map[*Name]*LogosRegionOwner),
	}
	logosRegionScan(fn.Body, &LogosRegionOwner{List: &fn.Body}, maps.owners, maps.decls)
	v, _ := logosRegionMapsCache.LoadOrStore(fn, maps)
	return v.(*logosRegionMaps)
}

func LogosRegionOwners(fn *Func) map[*CallExpr]*LogosRegionOwner {
	if fn == nil {
		return nil
	}
	return logosRegionMapsOf(fn).owners
}

func LogosRegionDeclOwner(fn *Func) map[*Name]*LogosRegionOwner {
	if fn == nil {
		return nil
	}
	return logosRegionMapsOf(fn).decls
}

func logosRegionScan(stmts []Node, owner *LogosRegionOwner, m map[*CallExpr]*LogosRegionOwner, decls map[*Name]*LogosRegionOwner) {
	for _, st := range stmts {
		logosRegionRecordDecls(st, owner, decls)
		if cl, ok := st.(*ClosureExpr); ok && cl.Func != nil {
			root := &LogosRegionOwner{List: &cl.Func.Body}
			logosRegionScan(cl.Func.Body, root, m, decls)
			continue
		}
		switch st.Op() {
		case OBLOCK:
			blk := st.(*BlockStmt)
			inner := &LogosRegionOwner{Stmt: blk, List: &blk.List, Parent: owner}
			logosRegionScan(blk.List, inner, m, decls)

		case OFOR:
			f := st.(*ForStmt)
			inner := owner
			if len(f.Init()) > 0 {
				inner = &LogosRegionOwner{Stmt: f, Parent: owner}
				logosRegionScan(f.Init(), inner, m, decls)
			}
			body := &LogosRegionOwner{Stmt: f, List: &f.Body, Parent: inner, Loop: true}
			logosRegionScan(f.Body, body, m, decls)
			logosRegionScanExprs([]Node{f.Cond, f.Post}, f, inner, m)

		case ORANGE:
			rg := st.(*RangeStmt)
			logosRegionScanExprs([]Node{rg.Key, rg.Value, rg.X}, rg, owner, m)
			body := &LogosRegionOwner{Stmt: rg, List: &rg.Body, Parent: owner, Loop: true}
			logosRegionScan(rg.Body, body, m, decls)

		case OIF:
			i := st.(*IfStmt)
			inner := owner
			if len(i.Init()) > 0 {
				inner = &LogosRegionOwner{Stmt: i, Parent: owner}
				logosRegionScan(i.Init(), inner, m, decls)
			}
			then := &LogosRegionOwner{Stmt: i, List: &i.Body, Parent: inner}
			logosRegionScan(i.Body, then, m, decls)
			if len(i.Else) > 0 {
				els := &LogosRegionOwner{Stmt: i, List: &i.Else, Parent: inner}
				logosRegionScan(i.Else, els, m, decls)
			}
			logosRegionScanExprs([]Node{i.Cond}, i, inner, m)

		case OSWITCH:
			sw := st.(*SwitchStmt)
			inner := owner
			if len(sw.Init()) > 0 {
				inner = &LogosRegionOwner{Stmt: sw, Parent: owner}
				logosRegionScan(sw.Init(), inner, m, decls)
			}
			logosRegionScanExprs([]Node{sw.Tag}, sw, inner, m)
			for _, c := range sw.Cases {
				if c == nil {
					continue
				}
				logosRegionScanExprs(c.List, c, inner, m)
				caseOwner := &LogosRegionOwner{Stmt: c, List: &c.Body, Parent: inner}
				logosRegionScan(c.Body, caseOwner, m, decls)
			}

		case OSELECT:
			sel := st.(*SelectStmt)
			for _, cc := range sel.Cases {
				if cc == nil {
					continue
				}
				if cc.Comm != nil {
					commOwner := &LogosRegionOwner{Stmt: cc.Comm, Parent: owner}
					logosRegionRecordDecls(cc.Comm, commOwner, decls)
					logosRegionScanNode(cc.Comm, commOwner, m)
				}
				caseOwner := &LogosRegionOwner{Stmt: cc, List: &cc.Body, Parent: owner}
				logosRegionScan(cc.Body, caseOwner, m, decls)
			}

		default:
			logosRegionScanNode(st, owner, m)
		}
	}
}

func logosRegionScanExprs(exprs []Node, stmt Node, owner *LogosRegionOwner, m map[*CallExpr]*LogosRegionOwner) {
	for _, e := range exprs {
		if !logosExprHasOpen(e) {
			continue
		}
		stmtOwner := &LogosRegionOwner{Stmt: stmt, Parent: owner}
		logosRegionScanNode(e, stmtOwner, m)
	}
}

func logosRegionScanNode(n Node, owner *LogosRegionOwner, m map[*CallExpr]*LogosRegionOwner) {
	if n == nil {
		return
	}
	if call, ok := n.(*CallExpr); ok && LogosRegionCallOp(call) == LogosRegionOpen {
		if m != nil {
			if _, seen := m[call]; !seen {
				m[call] = owner
			}
		}
		return
	}
	if cl, ok := n.(*ClosureExpr); ok && cl.Func != nil {
		root := &LogosRegionOwner{List: &cl.Func.Body}
		logosRegionScan(cl.Func.Body, root, m, nil)
		return
	}
	DoChildren(n, func(c Node) bool {
		logosRegionScanNode(c, owner, m)
		return false
	})
}

func logosRegionRecordDecls(st Node, owner *LogosRegionOwner, decls map[*Name]*LogosRegionOwner) {
	if decls == nil {
		return
	}
	Visit(st, func(n Node) {
		name, ok := n.(*Name)
		if !ok || name.Class != PAUTO || name.Defn == nil || name.Defn != st {
			return
		}
		if _, seen := decls[name]; !seen {
			if owner.Stmt == nil {
				decls[name] = nil
			} else {
				decls[name] = owner
			}
		}
	})
}

type LogosRegionPlan struct {
	mu       sync.Mutex
	placed   map[*CallExpr]bool
	liveOpen map[*CallExpr]bool
}

func (p *LogosRegionPlan) Place(call *CallExpr) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.placed == nil {
		p.placed = make(map[*CallExpr]bool)
	}
	p.placed[call] = true
}

func (p *LogosRegionPlan) Placed(call *CallExpr) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.placed[call]
}

func (p *LogosRegionPlan) Live(open *CallExpr) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.liveOpen == nil {
		p.liveOpen = make(map[*CallExpr]bool)
	}
	p.liveOpen[open] = true
}

func (p *LogosRegionPlan) IsLive(open *CallExpr) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.liveOpen[open]
}

var LogosRegions LogosRegionPlan
