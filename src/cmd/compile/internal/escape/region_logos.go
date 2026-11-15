// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"fmt"
	"slices"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

type logosRegion struct {
	call     *ir.CallExpr
	param    *ir.Name
	fn       *ir.Func
	owner    *ir.LogosRegionOwner
	record   *location
	contents *location
}

type logosCand struct {
	call   *ir.CallExpr
	fn     *ir.Func
	region *logosRegion
	loc    *location
}

type logosKill struct {
	desc    string
	closure bool
}

type logosRegionBatch struct {
	regions       map[*ir.CallExpr]*logosRegion
	sites         map[*ir.CallExpr]*logosCand
	candLocs      map[*location]*logosCand
	locs          map[*location]*logosRegion
	kills         map[*location]*logosKill
	flowDesc      map[*location]string
	contentsAll   []*location
	ensured       map[*ir.Func]bool
	facts         map[*ir.Func]*logosProvFacts
	groups        map[*location]*logosGroup
	groupOf       map[*ir.Name]*logosGroup
	groupMembers  map[*location]*logosGroup
	groupClosures map[*ir.ClosureExpr]bool
	groupConstOK  map[*ir.ClosureExpr]bool
	groupDead     map[*logosRegion]bool
}

type logosGroup struct {
	recv  *ir.Name
	loc   *location
	owner *ir.LogosRegionOwner
}

type logosProvFacts struct {
	prov        map[*ir.Name]*logosRegion
	paramRegion map[*ir.Name]*logosRegion
}

type LogosRegionFallback struct {
	Pos     src.XPos
	Message string
}

var logosRegionFallbacks []LogosRegionFallback

func LogosRegionFallbacks() []LogosRegionFallback {
	return logosRegionFallbacks
}

func logosRegionTypeOf(t *types.Type) bool {
	if t == nil || t.Sym() == nil || t.Sym().Pkg == nil {
		return false
	}
	return t.Sym().Pkg.Path == "logos/arena" && t.Sym().Name == "Region"
}

func logosRegionTransparentValue(call *ir.CallExpr) (ir.Node, bool) {
	if call == nil || call.Fun == nil {
		return nil, false
	}
	callee := ir.StaticCalleeName(call.Fun)
	if callee == nil {
		return nil, false
	}
	sym := callee.Sym()
	if sym == nil || sym.Pkg == nil || sym.Pkg.Path != "logos/rt" {
		return nil, false
	}
	name := sym.Name
	if i := strings.IndexByte(name, '['); i >= 0 {
		name = name[:i]
	}
	if name != "MustNonNil" {
		return nil, false
	}
	idx := ir.LogosRegionArgBase(call)
	if idx < 0 || idx >= len(call.Args) {
		return nil, false
	}
	return call.Args[idx], true
}

func logosRegionProvName(n ir.Node) (*ir.Name, bool) {
	for {
		if name, ok := n.(*ir.Name); ok {
			return name, true
		}
		if p, ok := n.(*ir.ParenExpr); ok {
			n = p.X
			continue
		}
		call, ok := n.(*ir.CallExpr)
		if !ok {
			return nil, false
		}
		value, ok := logosRegionTransparentValue(call)
		if !ok {
			return nil, false
		}
		n = value
	}
}

func logosRegionSinkName(n ir.Node) (*ir.Name, bool) {
	for {
		if name, ok := n.(*ir.Name); ok {
			return name, true
		}
		if addr, ok := n.(*ir.AddrExpr); ok {
			name, isName := addr.X.(*ir.Name)
			if !isName {
				return nil, false
			}
			return name, true
		}
		if p, ok := n.(*ir.ParenExpr); ok {
			n = p.X
			continue
		}
		call, ok := n.(*ir.CallExpr)
		if !ok {
			return nil, false
		}
		value, ok := logosRegionTransparentValue(call)
		if !ok {
			return nil, false
		}
		n = value
	}
}

func (b *batch) logosRegionInit() {
	if b.logosRegion.regions != nil {
		return
	}
	b.logosRegion.regions = make(map[*ir.CallExpr]*logosRegion)
	b.logosRegion.sites = make(map[*ir.CallExpr]*logosCand)
	b.logosRegion.candLocs = make(map[*location]*logosCand)
	b.logosRegion.locs = make(map[*location]*logosRegion)
	b.logosRegion.kills = make(map[*location]*logosKill)
	b.logosRegion.flowDesc = make(map[*location]string)
	b.logosRegion.ensured = make(map[*ir.Func]bool)
	b.logosRegion.facts = make(map[*ir.Func]*logosProvFacts)
	b.logosRegion.groups = make(map[*location]*logosGroup)
	b.logosRegion.groupOf = make(map[*ir.Name]*logosGroup)
	b.logosRegion.groupMembers = make(map[*location]*logosGroup)
	b.logosRegion.groupClosures = make(map[*ir.ClosureExpr]bool)
	b.logosRegion.groupConstOK = make(map[*ir.ClosureExpr]bool)
	b.logosRegion.groupDead = make(map[*logosRegion]bool)
}

func (b *batch) logosRegionEnsure(fn *ir.Func) {
	if b.logosRegion.ensured[fn] {
		return
	}
	b.logosRegionInit()
	b.logosRegion.ensured[fn] = true
	if !ir.LogosRegionsEnabled() {
		return
	}
	owners := ir.LogosRegionOwners(fn)
	ir.Visit(fn, func(n ir.Node) {
		call, ok := n.(*ir.CallExpr)
		if !ok {
			return
		}
		switch ir.LogosRegionCallOp(call) {
		case ir.LogosRegionOpen:
			if _, seen := b.logosRegion.regions[call]; seen {
				return
			}
			reg := &logosRegion{call: call, fn: fn, owner: owners[call]}
			reg.record = &location{n: call, curfn: fn, loopDepth: 1}
			reg.contents = b.logosRegionContentsLoc(call, fn)
			b.logosRegion.locs[reg.record] = reg
			b.logosRegion.locs[reg.contents] = reg
			b.logosRegion.regions[call] = reg
		case ir.LogosRegionNew, ir.LogosRegionMake, ir.LogosRegionAppend:
			if _, seen := b.logosRegion.sites[call]; !seen {
				b.logosRegion.sites[call] = &logosCand{call: call, fn: fn}
			}
		}
	})
}

func (b *batch) logosRegionParamRegion(facts *logosProvFacts, fn *ir.Func, name *ir.Name) *logosRegion {
	if reg, ok := facts.paramRegion[name.Canonical()]; ok {
		return reg
	}
	reg := &logosRegion{param: name, fn: fn}
	reg.contents = b.logosRegionContentsLoc(name, fn)
	b.logosRegion.locs[reg.contents] = reg
	facts.paramRegion[name.Canonical()] = reg
	return reg
}

func (b *batch) logosRegionContentsLoc(n ir.Node, fn *ir.Func) *location {
	loc := &location{n: n, curfn: fn, loopDepth: 1}
	b.logosRegion.contentsAll = append(b.logosRegion.contentsAll, loc)
	return loc
}

func (b *batch) logosRegionContents(reg *logosRegion) *location {
	if reg.contents == nil {
		n := ir.Node(reg.call)
		if reg.call == nil {
			n = reg.param
		}
		reg.contents = b.logosRegionContentsLoc(n, reg.fn)
		b.logosRegion.locs[reg.contents] = reg
	}
	return reg.contents
}

func (b *batch) logosRegionProv(fn *ir.Func) *logosProvFacts {
	if f, ok := b.logosRegion.facts[fn]; ok {
		return f
	}
	facts := b.logosRegionCompute(fn)
	b.logosRegion.facts[fn] = facts
	return facts
}

func (b *batch) logosRegionCompute(fn *ir.Func) *logosProvFacts {
	facts := &logosProvFacts{prov: make(map[*ir.Name]*logosRegion), paramRegion: make(map[*ir.Name]*logosRegion)}
	if !ir.LogosRegionsEnabled() {
		return facts
	}
	b.logosRegionInit()
	d := b.logosRegionDefs(fn)
	for _, f := range fn.Type().RecvParams() {
		name, ok := f.Nname.(*ir.Name)
		if !ok || name.Class != ir.PPARAM || !logosRegionTypeOf(name.Type()) {
			continue
		}
		reg := b.logosRegionParamRegion(facts, fn, name)
		facts.paramRegion[name.Canonical()] = reg
		facts.prov[name.Canonical()] = reg
		if _, hasDefs := d.defs[name.Canonical()]; hasDefs {
			d.defs[name.Canonical()] = append([]ir.Node{name}, d.defs[name.Canonical()]...)
		}
	}
	for name := range d.captured {
		d.tainted[name] = true
	}
	pending := make(map[*ir.Name]bool, len(d.defs))
	for name := range d.defs {
		if !d.tainted[name] {
			pending[name] = true
		}
	}
	for range len(d.defs) + 1 {
		changed := false
		for name, rhss := range d.defs {
			if d.tainted[name] || !pending[name] {
				continue
			}
			var r *logosRegion
			state := logosRhsOK
			for _, rhs := range rhss {
				rr, st := b.logosRegionRhs(rhs, facts.prov, pending)
				if st == logosRhsBad {
					state = logosRhsBad
					break
				}
				if st == logosRhsPending {
					state = logosRhsPending
					continue
				}
				if rr == nil {
					continue
				}
				if r == nil {
					r = rr
				} else if r != rr {
					state = logosRhsBad
					break
				}
			}
			switch state {
			case logosRhsBad:
				d.tainted[name] = true
				delete(facts.prov, name)
				pending[name] = false
				for other := range pending {
					if !d.tainted[other] {
						pending[other] = true
					}
				}
				changed = true
			case logosRhsOK:
				pending[name] = false
				if facts.prov[name] != r {
					facts.prov[name] = r
					changed = true
				}
			}
		}
		if !changed {
			break
		}
	}
	for name := range d.tainted {
		delete(facts.prov, name)
	}
	return facts
}

func (b *batch) logosRegionDefs(fn *ir.Func) *logosDefs {
	d := &logosDefs{
		defs:     make(map[*ir.Name][]ir.Node),
		tainted:  make(map[*ir.Name]bool),
		captured: make(map[*ir.Name]bool),
	}
	var scan func(n ir.Node)
	scan = func(n ir.Node) {
		switch n := n.(type) {
		case *ir.AssignStmt:
			if dst, ok := n.X.(*ir.Name); ok && dst.Curfn == fn {
				if n.Op() == ir.OAS {
					d.defs[dst.Canonical()] = append(d.defs[dst.Canonical()], n.Y)
				} else if n.Op() == ir.OASOP {
					d.tainted[dst.Canonical()] = true
				}
			}
		case *ir.AssignListStmt:
			switch n.Op() {
			case ir.OAS2:
				for i, x := range n.Lhs {
					if dst, ok := x.(*ir.Name); ok && dst.Curfn == fn && i < len(n.Rhs) {
						d.defs[dst.Canonical()] = append(d.defs[dst.Canonical()], n.Rhs[i])
					}
				}
			case ir.OAS2FUNC, ir.OAS2DOTTYPE, ir.OAS2MAPR, ir.OAS2RECV:
				for _, x := range n.Lhs {
					if dst, ok := x.(*ir.Name); ok && dst.Curfn == fn {
						d.tainted[dst.Canonical()] = true
					}
				}
			}
		case *ir.RangeStmt:
			for _, v := range []ir.Node{n.Key, n.Value} {
				if name, ok := v.(*ir.Name); ok && name.Curfn == fn {
					d.tainted[name.Canonical()] = true
				}
			}
		case *ir.AddrExpr:
			if x, ok := n.X.(*ir.Name); ok && x.Curfn == fn {
				d.captured[x.Canonical()] = true
			}
		case *ir.ClosureExpr:
			if n.Func != nil {
				for _, cv := range n.Func.ClosureVars {
					d.captured[cv.Canonical()] = true
				}
			}
			return
		}
		ir.DoChildren(n, func(c ir.Node) bool {
			scan(c)
			return false
		})
	}
	for _, st := range fn.Body {
		scan(st)
	}
	return d
}

type logosDefs struct {
	defs     map[*ir.Name][]ir.Node
	tainted  map[*ir.Name]bool
	captured map[*ir.Name]bool
}

type logosRhsState uint8

const (
	logosRhsBad logosRhsState = iota
	logosRhsPending
	logosRhsOK
)

func (b *batch) logosRegionRhs(rhs ir.Node, prov map[*ir.Name]*logosRegion, pending map[*ir.Name]bool) (*logosRegion, logosRhsState) {
	switch n := rhs.(type) {
	case nil:
		return nil, logosRhsOK
	case *ir.NilExpr:
		return nil, logosRhsOK
	case *ir.Name:
		return b.logosRegionProvOf(n, prov, pending)
	case *ir.CallExpr:
		if value, ok := logosRegionTransparentValue(n); ok {
			return b.logosRegionProvOf(value, prov, pending)
		}
		switch ir.LogosRegionCallOp(n) {
		case ir.LogosRegionOpen:
			r := b.logosRegion.regions[n]
			if r == nil {
				return nil, logosRhsBad
			}
			return r, logosRhsOK
		case ir.LogosRegionNew, ir.LogosRegionMake:
			return b.logosRegionProvOf(n.Args[ir.LogosRegionArgBase(n)], prov, pending)
		case ir.LogosRegionAppend:
			base := ir.LogosRegionArgBase(n)
			r, st := b.logosRegionProvOf(n.Args[base], prov, pending)
			if st != logosRhsOK {
				return nil, st
			}
			if _, ok := n.Args[base+1].(*ir.NilExpr); ok {
				return r, logosRhsOK
			}
			if s, ok := logosRegionProvName(n.Args[base+1]); ok {
				if prov[s.Canonical()] == r {
					return r, logosRhsOK
				}
				if pending[s.Canonical()] {
					return nil, logosRhsPending
				}
			}
			return nil, logosRhsBad
		}
	}
	return nil, logosRhsBad
}

func (b *batch) logosRegionProvOf(n ir.Node, prov map[*ir.Name]*logosRegion, pending map[*ir.Name]bool) (*logosRegion, logosRhsState) {
	for n.Op() == ir.OPAREN {
		n = n.(*ir.ParenExpr).X
	}
	if name, ok := logosRegionProvName(n); ok {
		if r, ok := prov[name.Canonical()]; ok {
			return r, logosRhsOK
		}
		if pending[name.Canonical()] {
			return nil, logosRhsPending
		}
		return nil, logosRhsBad
	}
	if call, ok := n.(*ir.CallExpr); ok && ir.LogosRegionCallOp(call) == ir.LogosRegionOpen {
		if r := b.logosRegion.regions[call]; r != nil {
			return r, logosRhsOK
		}
	}
	return nil, logosRhsBad
}

func (b *batch) logosRegionOf(n ir.Node, prov map[*ir.Name]*logosRegion) *logosRegion {
	for n.Op() == ir.OPAREN {
		n = n.(*ir.ParenExpr).X
	}
	switch m := n.(type) {
	case *ir.CallExpr:
		if ir.LogosRegionCallOp(m) == ir.LogosRegionOpen {
			return b.logosRegion.regions[m]
		}
	case *ir.Name:
		return prov[m.Canonical()]
	}
	return nil
}

func (e *escape) logosRegionCall(ks []hole, call *ir.CallExpr) {
	if !ir.LogosRegionsEnabled() || len(ks) == 0 {
		return
	}
	e.logosRegionEnsure(e.curfn)
	op := ir.LogosRegionCallOp(call)
	switch op {
	case ir.LogosRegionOpen:
		reg := e.logosRegion.regions[call]
		if reg != nil && ks[0].dst != &e.blankLoc {
			e.flow(ks[0].addr(call, "region record"), reg.record)
		}
	case ir.LogosRegionNew, ir.LogosRegionMake, ir.LogosRegionAppend:
		site := e.logosRegion.sites[call]
		if site == nil {
			return
		}
		prov := e.logosRegionProv(e.curfn)
		argBase := ir.LogosRegionArgBase(call)
		r := e.logosRegionOf(call.Args[argBase], prov.prov)
		if r == nil {
			return
		}
		if op == ir.LogosRegionAppend {
			if _, ok := call.Args[argBase+1].(*ir.NilExpr); !ok {
				name, ok := logosRegionProvName(call.Args[argBase+1])
				if !ok || prov.prov[name.Canonical()] != r {
					return
				}
			}
		}
		loc := &location{n: call, curfn: e.curfn, loopDepth: e.loopDepth}
		site.region = r
		site.loc = loc
		e.logosRegion.candLocs[loc] = site
		e.flow(ks[0].addr(call, "region alloc"), loc)
		if op == ir.LogosRegionAppend {
			e.exprSkipInit(e.logosRegionContents(r).asHole().note(call, "region append"), call.Args[argBase+2])
		}
	}
}

func (b *batch) logosRegionSeed(enqueue func(*location)) {
	if !ir.LogosRegionsEnabled() {
		return
	}
	for _, loc := range b.logosRegion.contentsAll {
		enqueue(loc)
	}
}

func (b *batch) logosRegionOutlives(l, other *location) (bool, bool) {
	if !ir.LogosRegionsEnabled() || len(b.logosRegion.candLocs) == 0 {
		return false, false
	}
	if g := b.logosRegionGroupLoc(l); g != nil {
		if site, ok := b.logosRegion.candLocs[other]; ok {
			return b.logosRegionGroupOutlives(g, site), true
		}
		if _, ok := b.logosRegion.locs[other]; ok {
			return true, true
		}
		return false, true
	}
	if site, ok := b.logosRegion.candLocs[other]; ok {
		return b.logosRegionRootOutlives(l, site), true
	}
	if reg, ok := b.logosRegion.locs[l]; ok && reg.contents == l {
		return true, true
	}
	return false, false
}

func (b *batch) logosRegionRootOutlives(root *location, site *logosCand) bool {
	if root == &b.heapLoc {
		return true
	}
	if root == &b.foreignLoc {
		return false
	}
	if root.hasAttr(attrEscapes) {
		return true
	}
	name := logosRegionLocName(root)
	if name != nil {
		if name.Class == ir.PEXTERN || root.paramOut {
			return true
		}
		if root.param {
			return false
		}
		reg := site.region
		if reg == nil {
			return true
		}
		if reg.owner == nil {
			return name.Curfn != site.fn
		}
		if name.Curfn != reg.fn {
			if name.Curfn != nil && b.logosGroupConstParam(name) {
				return false
			}
			return true
		}
		declOwner := ir.LogosRegionDeclOwner(name.Curfn)[name]
		if declOwner == nil && name.Canonical() != name {
			declOwner = ir.LogosRegionDeclOwner(name.Curfn)[name.Canonical()]
		}
		if declOwner == nil {
			return !(reg.owner.Parent == nil && reg.owner.List != nil && reg.owner.List == &reg.fn.Body)
		}
		return !reg.owner.Encloses(declOwner)
	}
	if reg, ok := b.logosRegion.locs[root]; ok && reg.contents == root {
		if reg == site.region {
			return false
		}
		if reg.owner != nil && site.region != nil && site.region.owner != nil {
			return !site.region.owner.Encloses(reg.owner)
		}
		return true
	}
	return false
}

func (b *batch) logosGroupConstParam(name *ir.Name) bool {
	if LogosConstViewDecl == nil || !LogosConstViewDecl(name.Canonical().Pos()) {
		return false
	}
	clo := name.Curfn.OClosure
	return clo != nil && b.logosRegion.groupConstOK[clo]
}

func logosRegionLocName(l *location) *ir.Name {
	if l == nil || l.n == nil {
		return nil
	}
	if name, ok := l.n.(*ir.Name); ok && name.Op() == ir.ONAME {
		return name
	}
	return nil
}

func (b *batch) logosRegionEscaped(l, root *location) {
	if !ir.LogosRegionsEnabled() {
		return
	}
	known := false
	if _, ok := b.logosRegion.candLocs[l]; ok {
		known = true
	} else if reg, ok := b.logosRegion.locs[l]; ok && reg.record == l {
		known = true
	}
	if !known {
		return
	}
	if _, done := b.logosRegion.kills[l]; done {
		return
	}
	k := &logosKill{desc: b.logosRegionKillDesc(l, root)}
	if reg, ok := b.logosRegion.locs[l]; ok && reg.record == l {
		k.closure = b.logosRegionChainClosure(l, root)
	}
	b.logosRegion.kills[l] = k
}

func (b *batch) logosRegionKillDesc(l, root *location) string {
	if l.dst != nil {
		if desc, ok := b.logosRegion.flowDesc[l.dst]; ok {
			return "value reaches " + desc
		}
	}
	if reg, ok := b.logosRegion.locs[l]; ok && reg.record == l {
		return "region handle reaches " + b.explainLoc(root)
	}
	return "flows to " + b.explainLoc(root)
}

func (b *batch) logosRegionChainClosure(l, root *location) bool {
	for seen, d := 0, l.dst; d != nil && d != root && seen < 64; d, seen = d.dst, seen+1 {
		if d.n != nil && d.n.Op() == ir.OCLOSURE {
			return true
		}
	}
	return false
}
func (e *escape) logosRegionRecordStores(dsts, srcs []ir.Node, ks []hole) {
	e.logosRegionEnsure(e.curfn)
	if len(e.logosRegion.regions) == 0 && len(e.logosRegion.sites) == 0 {
		return
	}
	for i := range dsts {
		if i >= len(srcs) {
			return
		}
		desc := logosRegionStoreDesc(dsts[i])
		if desc == "" {
			continue
		}
		switch src := srcs[i].(type) {
		case *ir.Name:
			if src.Canonical().Opt == nil {
				continue
			}
			e.logosRegion.flowDesc[e.oldLoc(src.Canonical())] = desc
		case *ir.CallExpr:
			if ir.LogosRegionCallOp(src) != ir.LogosRegionNone && i < len(ks) && ks[i].dst != nil {
				e.logosRegion.flowDesc[ks[i].dst] = desc
			}
		}
	}
}

func logosRegionStoreDesc(dst ir.Node) string {
	for {
		switch n := dst.(type) {
		case *ir.ParenExpr:
			dst = n.X
		case *ir.Name:
			if n.Op() != ir.ONAME || n.Sym() == nil || n.Sym().IsBlank() {
				return ""
			}
			if n.Class == ir.PEXTERN {
				return "global"
			}
			return fmt.Sprintf("%v", n)
		case *ir.SelectorExpr:
			dst = n.X
		case *ir.StarExpr:
			dst = n.X
		case *ir.IndexExpr:
			dst = n.X
		default:
			return ""
		}
	}
}

func (e *escape) logosRegionRecordArg(arg ir.Node, fnName *ir.Name) {
	if !ir.LogosRegionsEnabled() || fnName == nil {
		return
	}
	name, ok := logosRegionSinkName(arg)
	if !ok || name.Canonical().Opt == nil {
		return
	}
	e.logosRegionEnsure(e.curfn)
	e.logosRegion.flowDesc[e.oldLoc(name.Canonical())] = "call to " + fnName.Sym().Name
}

func (e *escape) logosRegionStoreHole(dst ir.Node) (hole, bool) {
	if !ir.LogosRegionsEnabled() {
		return hole{}, false
	}
	switch dst.(type) {
	case *ir.SelectorExpr, *ir.StarExpr, *ir.IndexExpr:
	default:
		return hole{}, false
	}
	name := logosRegionStoreRoot(dst)
	if name == nil {
		return hole{}, false
	}
	reg := e.logosRegionProv(e.curfn).prov[name.Canonical()]
	if reg == nil {
		return hole{}, false
	}
	return e.logosRegionContents(reg).asHole(), true
}

func logosRegionStoreRoot(dst ir.Node) *ir.Name {
	for {
		switch n := dst.(type) {
		case *ir.ParenExpr:
			dst = n.X
		case *ir.Name:
			if n.Op() != ir.ONAME || n.Class == ir.PEXTERN || n.Sym() == nil || n.Sym().IsBlank() {
				return nil
			}
			return n
		case *ir.SelectorExpr:
			dst = n.X
		case *ir.StarExpr:
			dst = n.X
		case *ir.IndexExpr:
			dst = n.X
		default:
			return nil
		}
	}
}

func (e *escape) logosRegionArgHole(base ir.Node) (hole, bool) {
	if !ir.LogosRegionsEnabled() {
		return hole{}, false
	}
	name, ok := logosRegionSinkName(base)
	if !ok {
		return hole{}, false
	}
	reg := e.logosRegionProv(e.curfn).prov[name.Canonical()]
	if reg == nil {
		return hole{}, false
	}
	return e.logosRegionContents(reg).asHole(), true
}
func logosGroupSplit(call *ir.CallExpr, recvArg ir.Node, args ir.Nodes) (*ir.Name, *ir.ClosureExpr, bool) {
	var recvN ir.Node
	var litN ir.Node
	if recvArg != nil {
		if len(args) != 1 {
			return nil, nil, false
		}
		recvN, litN = recvArg, args[0]
	} else {
		if len(args) != 2 {
			return nil, nil, false
		}
		recvN, litN = args[0], args[1]
	}
	recv, ok := recvN.(*ir.Name)
	if !ok || recv.Sym() == nil || !strings.HasPrefix(recv.Sym().Name, "_logos_g") {
		return nil, nil, false
	}
	lit, ok := litN.(*ir.ClosureExpr)
	if !ok {
		return nil, nil, false
	}
	return recv, lit, true
}

func (e *escape) logosGroupGo(call *ir.CallExpr, recvArg ir.Node, args ir.Nodes) bool {
	if !ir.LogosRegionsEnabled() || !base.Logos || call.Op() != ir.OCALLFUNC {
		return false
	}
	sel, ok := call.Fun.(*ir.SelectorExpr)
	if !ok || sel.Sel == nil || sel.Sel.Name != "Go" {
		return false
	}
	recv, _, ok := logosGroupSplit(call, recvArg, args)
	if !ok {
		return false
	}
	if recv.Curfn != e.curfn {
		return false
	}
	t := recv.Type()
	if t == nil || !t.IsPtr() {
		return false
	}
	sym := t.Elem().Sym()
	if sym == nil || sym.Name != "Group" || sym.Pkg == nil || sym.Pkg.Path != "logos/rt" {
		return false
	}
	return true
}

func (e *escape) logosGroupGoArgs(call *ir.CallExpr, recvArg ir.Node, args ir.Nodes, argument func(hole, ir.Node)) {
	e.expr(e.discardHole(), call.Fun)
	recv, lit, ok := logosGroupSplit(call, recvArg, args)
	if !ok {
		base.FatalfAt(call.Pos(), "logos: group Go shape changed")
	}
	e.discard(recv)
	recv = recv.Canonical()
	e.logosRegionEnsure(e.curfn)
	g := e.logosRegion.groupOf[recv]
	if g == nil {
		g = &logosGroup{recv: recv, loc: e.newLoc(nil, false)}
		g.owner = ir.LogosRegionDeclOwner(e.curfn)[recv]
		e.logosRegion.groupOf[recv] = g
		e.logosRegion.groups[g.loc] = g
	}
	n0 := len(e.allLocs)
	argument(g.loc.asHole().note(call, "group child"), lit)
	var clo *location
	for _, loc := range e.allLocs[n0:] {
		if loc.n == ir.Node(lit) {
			clo = loc
			break
		}
	}
	if clo != nil {
		clo.attrs |= attrEscapes
		e.logosRegion.groupMembers[clo] = g
		e.logosRegion.groupClosures[lit] = true
	}
	for _, cv := range lit.Func.ClosureVars {
		vc := cv.Canonical()
		if reg := e.logosRegionProv(e.curfn).prov[vc]; reg != nil {
			e.logosRegion.groupDead[reg] = true
		}
		if e.logosGroupConstRead(e.curfn, lit, vc) {
			e.logosRegion.groupConstOK[lit] = true
			continue
		}
		loc := e.oldLoc(vc)
		e.flow(e.heapHole().note(call, "group child"), loc)
		e.logosRegion.flowDesc[loc] = "group child"
		if src := logosGroupTempSource(vc); src != nil {
			if opt, ok := src.Canonical().Opt.(*location); ok {
				e.logosRegion.flowDesc[opt] = "group child"
			}
		}
	}
}

func (b *batch) logosGroupConstRead(fn *ir.Func, lit *ir.ClosureExpr, vc *ir.Name) bool {
	if LogosConstViewDecl == nil || !LogosConstViewDecl(vc.Pos()) {
		return false
	}
	if sym := vc.Sym(); sym != nil && strings.HasPrefix(sym.Name, "_logos_") {
		if !b.logosGroupConstCopy(lit, vc) {
			return false
		}
	}
	return b.logosGroupConstLifetimeOK(fn, vc)
}

func (b *batch) logosGroupConstCopy(lit *ir.ClosureExpr, vc *ir.Name) bool {
	found := false
	ir.VisitList(lit.Func.Body, func(n ir.Node) {
		if found {
			return
		}
		switch n := n.(type) {
		case *ir.AssignStmt:
			if n.Op() != ir.OAS || n.Y == nil {
				return
			}
			x, ok := n.X.(*ir.Name)
			if !ok || logosGroupTempSource(x.Canonical()) != vc {
				return
			}
			if LogosConstViewDecl != nil && LogosConstViewDecl(x.Canonical().Pos()) {
				found = true
			}
		case *ir.AssignListStmt:
			if n.Op() != ir.OAS2 {
				return
			}
			for i, lhs := range n.Lhs {
				x, ok := lhs.(*ir.Name)
				if !ok || i >= len(n.Rhs) {
					continue
				}
				if logosGroupTempSource(x.Canonical()) != vc {
					continue
				}
				if LogosConstViewDecl != nil && LogosConstViewDecl(x.Canonical().Pos()) {
					found = true
					return
				}
			}
		case *ir.CallExpr:
			if n.Op() != ir.OCALLFUNC && n.Op() != ir.OCALLINTER {
				return
			}
			ft := n.Fun.Type()
			if ft == nil || ft.Recv() != nil {
				return
			}
			params := ft.Params()
			for i, arg := range n.Args {
				if i >= len(params) {
					break
				}
				y, ok := logosRegionProvName(arg)
				if !ok || y.Canonical() != vc {
					continue
				}
				if name, ok := params[i].Nname.(*ir.Name); ok && LogosConstViewDecl != nil && LogosConstViewDecl(name.Pos()) {
					found = true
					return
				}
			}
		}
	})
	return found
}

func (b *batch) logosGroupConstLifetimeOK(fn *ir.Func, vc *ir.Name) bool {
	reg := b.logosGroupTempRegion(fn, vc)
	if reg == nil || reg.fn != fn || reg.owner == nil {
		return false
	}
	useOwner := ir.LogosRegionDeclOwner(fn)[vc]
	if useOwner == nil && vc.Canonical() != vc {
		useOwner = ir.LogosRegionDeclOwner(fn)[vc.Canonical()]
	}
	if useOwner == nil {
		return reg.owner.Parent == nil && reg.owner.List != nil && reg.owner.List == &reg.fn.Body
	}
	return reg.owner.Encloses(useOwner)
}

func (b *batch) logosGroupTempRegion(fn *ir.Func, vc *ir.Name) *logosRegion {
	seen := make(map[*ir.Name]bool)
	src := vc
	for {
		if seen[src] {
			return nil
		}
		seen[src] = true
		if reg := b.logosRegionProv(fn).prov[src]; reg != nil {
			return reg
		}
		rhs := logosGroupBoundValue(src)
		if rhs == nil {
			return nil
		}
		if name, ok := logosRegionProvName(rhs); ok && name.Canonical() != src {
			src = name.Canonical()
			continue
		}
		if cl, ok := rhs.(*ir.CompLitExpr); ok {
			return b.logosGroupCompositeRegion(fn, cl)
		}
		base, ok := logosGroupElemName(rhs)
		if !ok || base == src || seen[base] {
			return nil
		}
		src = base
	}
}

func logosGroupBoundValue(src *ir.Name) ir.Node {
	switch d := src.Defn.(type) {
	case *ir.AssignStmt:
		if d.Op() != ir.OAS || d.Y == nil {
			return nil
		}
		if x, ok := d.X.(*ir.Name); !ok || x.Canonical() != src {
			return nil
		}
		return d.Y
	case *ir.AssignListStmt:
		if d.Op() != ir.OAS2 {
			return nil
		}
		for i, lhs := range d.Lhs {
			if n, ok := lhs.(*ir.Name); ok && n.Canonical() == src && i < len(d.Rhs) {
				return d.Rhs[i]
			}
		}
	}
	return nil
}

func (b *batch) logosGroupCompositeRegion(fn *ir.Func, cl *ir.CompLitExpr) *logosRegion {
	var reg *logosRegion
	for _, elem := range cl.List {
		val := elem
		if kv, ok := val.(*ir.KeyExpr); ok {
			val = kv.Value
		}
		name, ok := logosGroupElemName(val)
		if !ok {
			return nil
		}
		r := b.logosRegionProv(fn).prov[name]
		if r == nil {
			return nil
		}
		if reg == nil {
			reg = r
		} else if reg != r {
			return nil
		}
	}
	return reg
}

func logosGroupElemName(n ir.Node) (*ir.Name, bool) {
	for {
		switch v := n.(type) {
		case *ir.ParenExpr:
			n = v.X
		case *ir.AddrExpr:
			n = v.X
		case *ir.IndexExpr:
			n = v.X
		case *ir.Name:
			return v.Canonical(), true
		case *ir.CallExpr:
			value, ok := logosRegionTransparentValue(v)
			if !ok {
				return nil, false
			}
			n = value
		default:
			return nil, false
		}
	}
}

func logosGroupTempSource(temp *ir.Name) *ir.Name {
	switch d := temp.Defn.(type) {
	case *ir.AssignStmt:
		if d.Op() != ir.OAS || d.Y == nil {
			return nil
		}
		if x, ok := d.X.(*ir.Name); ok && x.Canonical() == temp {
			if y, ok := logosRegionProvName(d.Y); ok {
				return y.Canonical()
			}
		}
	case *ir.AssignListStmt:
		if d.Op() != ir.OAS2 {
			return nil
		}
		for i, lhs := range d.Lhs {
			if n, ok := lhs.(*ir.Name); ok && n.Canonical() == temp && i < len(d.Rhs) {
				if y, ok := logosRegionProvName(d.Rhs[i]); ok {
					return y.Canonical()
				}
			}
		}
	}
	return nil
}
func (b *batch) logosRegionSkipGroupCapture(clo *ir.ClosureExpr, cv *ir.Name) bool {
	if !ir.LogosRegionsEnabled() || !base.Logos || !b.logosRegion.groupConstOK[clo] {
		return false
	}
	vc := cv.Canonical()
	fn := vc.Curfn
	if fn == nil {
		return false
	}
	return b.logosGroupConstRead(fn, clo, vc)
}

func (b *batch) logosRegionVeto(fn *ir.Func) string {
	if LogosArenaPlaceable(fn) {
		return ""
	}
	info := logosArenaInfo(fn)
	if info.hasGoto {
		return "function contains goto"
	}
	if info.hasDefer && b.logosRegionHasUserDefer(fn) {
		return "function contains defer"
	}
	if info.hasDefer {
		return ""
	}
	return "region unknown"
}

func (b *batch) logosRegionHasUserDefer(fn *ir.Func) bool {
	user := false
	var scan func(n ir.Node)
	scan = func(n ir.Node) {
		if user {
			return
		}
		switch n := n.(type) {
		case nil:
			return
		case *ir.ClosureExpr:
			return
		case *ir.GoDeferStmt:
			if n.Op() == ir.ODEFER && !b.logosRegionGuardDefer(n) {
				user = true
			}
			return
		}
		ir.DoChildren(n, func(c ir.Node) bool {
			scan(c)
			return false
		})
	}
	for _, st := range fn.Body {
		scan(st)
	}
	return user
}

func (b *batch) logosRegionGuardDefer(d *ir.GoDeferStmt) bool {
	found := false
	var scan func(n ir.Node)
	scan = func(n ir.Node) {
		if found || n == nil {
			return
		}
		if name, ok := n.(*ir.Name); ok {
			if name.Sym() != nil && strings.HasPrefix(name.Sym().Name, "_logos_g") {
				if b.logosRegion.groupOf[name.Canonical()] != nil {
					found = true
				}
			}
			return
		}
		if clo, ok := n.(*ir.ClosureExpr); ok && clo.Func != nil {
			scan(clo.Func)
			return
		}
		ir.DoChildren(n, func(c ir.Node) bool {
			scan(c)
			return false
		})
	}
	scan(d)
	return found
}
func (b *batch) logosRegionGroupLoc(l *location) *logosGroup {
	if g := b.logosRegion.groups[l]; g != nil {
		return g
	}
	return b.logosRegion.groupMembers[l]
}

func (b *batch) logosRegionGroupOutlives(g *logosGroup, site *logosCand) bool {
	reg := site.region
	if reg == nil || reg.owner == nil {
		return true
	}
	if g.owner == nil {
		return !(reg.owner.Parent == nil && reg.owner.List != nil && reg.owner.List == &reg.fn.Body)
	}
	return !reg.owner.Encloses(g.owner)
}

func (b *batch) logosRegionMark(fns []*ir.Func) {
	if !ir.LogosRegionsEnabled() {
		return
	}
	regCalls := make([]*ir.CallExpr, 0, len(b.logosRegion.regions))
	for call := range b.logosRegion.regions {
		regCalls = append(regCalls, call)
	}
	slices.SortFunc(regCalls, func(a, c *ir.CallExpr) int {
		return strings.Compare(base.FmtPos(a.Pos()), base.FmtPos(c.Pos()))
	})
	for _, call := range regCalls {
		reg := b.logosRegion.regions[call]
		if b.logosRegion.groupDead[reg] {
			b.logosRegionFallback("arena.Open", call.Pos(), "region handle reaches group child")
			continue
		}
		if !reg.record.hasAttr(attrEscapes) {
			ir.LogosRegions.Live(call)
			continue
		}
		k := b.logosRegion.kills[reg.record]
		if base.Logos && k != nil && k.closure {
			base.ErrorfAt(call.Pos(), 0, "[correctness::region_escape] region handle escapes through a closure")
			continue
		}
		desc := "region handle escapes"
		if k != nil && k.desc != "" {
			desc = k.desc
		}
		b.logosRegionFallback("arena.Open", call.Pos(), desc)
	}
	siteCalls := make([]*ir.CallExpr, 0, len(b.logosRegion.sites))
	for call := range b.logosRegion.sites {
		siteCalls = append(siteCalls, call)
	}
	slices.SortFunc(siteCalls, func(a, c *ir.CallExpr) int {
		return strings.Compare(base.FmtPos(a.Pos()), base.FmtPos(c.Pos()))
	})
	for _, call := range siteCalls {
		site := b.logosRegion.sites[call]
		if reason := b.logosRegionSiteReason(site); reason != "" {
			b.logosRegionFallback(logosRegionCallText(call), call.Pos(), reason)
			continue
		}
		ir.LogosRegions.Place(call)
		if base.Flag.LowerM != 0 {
			base.WarnfAt(call.Pos(), "%s", "placed in region")
		}
	}
}

func (b *batch) logosRegionSiteReason(site *logosCand) string {
	if reason := b.logosRegionVeto(site.fn); reason != "" {
		return reason
	}
	if site.fn.Dupok() || logosRegionShapedBody(site.fn) {
		return "shaped generic body"
	}
	if site.region == nil {
		return "region unknown"
	}
	if site.region.record != nil && site.region.record.hasAttr(attrEscapes) {
		if k := b.logosRegion.kills[site.region.record]; k != nil && k.desc != "" {
			return k.desc
		}
		return "region handle escapes"
	}
	if site.loc != nil && site.loc.hasAttr(attrEscapes) {
		if k := b.logosRegion.kills[site.loc]; k != nil && k.desc != "" {
			return k.desc
		}
		return "flows to " + b.explainLoc(&b.heapLoc)
	}
	elem := ir.LogosRegionElem(site.call)
	if elem == nil {
		return ""
	}
	switch ir.LogosRegionCallOp(site.call) {
	case ir.LogosRegionNew:
		if elem.Size() > LogosArenaMaxBytes {
			return "size exceeds region chunk"
		}
	case ir.LogosRegionMake:
		if n, ok := constInt(site.call.Args[ir.LogosRegionArgBase(site.call)+1]); ok && n >= 0 && int64(elem.Size())*n > LogosArenaMaxBytes {
			return "size exceeds region chunk"
		}
	}
	return ""
}

func logosRegionShapedBody(fn *ir.Func) bool {
	for _, f := range fn.Type().RecvParams() {
		if f.Type != nil && f.Type.IsShape() {
			return true
		}
	}
	return false
}

func logosRegionCallText(call *ir.CallExpr) string {
	name := "New"
	switch ir.LogosRegionCallOp(call) {
	case ir.LogosRegionOpen:
		name = "Open"
	case ir.LogosRegionMake:
		name = "Make"
	case ir.LogosRegionAppend:
		name = "Append"
	}
	text := "arena." + name
	if elem := ir.LogosRegionElem(call); elem != nil {
		text += "[" + elem.String() + "]"
	}
	return text
}

func (b *batch) logosRegionFallback(text string, pos src.XPos, reason string) {
	logosRegionFallbacks = append(logosRegionFallbacks, LogosRegionFallback{
		Pos:     pos,
		Message: fmt.Sprintf("%s at %s falls back to the heap: %s", text, base.FmtPos(pos), reason),
	})
	if base.Flag.LowerM != 0 {
		base.WarnfAt(pos, "%s", "region fallback: "+reason)
	}
}
