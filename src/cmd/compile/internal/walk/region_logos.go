// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package walk

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"go/constant"
	"internal/buildcfg"
	"slices"
)

func logosRegionLower(fn *ir.Func) {
	if fn == nil || !ir.LogosRegionsEnabled() || len(fn.Body) == 0 {
		return
	}
	p := logosRegionScan(fn)
	if !p.relevant {
		return
	}
	old := base.Pos
	defer func() { base.Pos = old }()
	root := p.funcOwner(&fn.Body)
	p.lowerOwned(&fn.Body, root)
	for _, e := range p.appends {
		ssagen.LogosRegionAppends.Store(e.app, ssagen.LogosRegionGrowInfo{H: e.h, G: e.g})
	}
}

type logosRegionOwnerKey struct {
	stmt ir.Node
	list *ir.Nodes
}

type logosRegionAppendEntry struct {
	app *ir.CallExpr
	h   ir.Node
	g   ir.Node
}

type logosRegionLowering struct {
	fn       *ir.Func
	live     map[logosRegionOwnerKey][]*ir.CallExpr
	names    map[*ir.CallExpr]*ir.Name
	sites    map[*ir.CallExpr]*ir.Name
	seq      map[*ir.CallExpr]int
	labels   map[*types.Sym]ir.Node
	init     *ir.Nodes
	appends  []logosRegionAppendEntry
	relevant bool
}

type logosRegionOwner struct {
	stmt      ir.Node
	parent    *logosRegionOwner
	loop      bool
	caseOwner bool
	clause    bool
	closables []*ir.CallExpr
}

func logosRegionScan(fn *ir.Func) *logosRegionLowering {
	p := &logosRegionLowering{
		fn:    fn,
		live:  make(map[logosRegionOwnerKey][]*ir.CallExpr),
		names: make(map[*ir.CallExpr]*ir.Name),
		sites: make(map[*ir.CallExpr]*ir.Name),
		seq:   make(map[*ir.CallExpr]int),
	}
	order := 0
	ir.VisitFuncAndClosures(fn, func(n ir.Node) {
		as, ok := n.(*ir.AssignStmt)
		if ok && as.Y != nil {
			if call, ok := as.Y.(*ir.CallExpr); ok && ir.LogosRegionCallOp(call) == ir.LogosRegionOpen {
				if nm, ok := as.X.(*ir.Name); ok {
					p.names[call] = nm
				}
			}
		}
		call, ok := n.(*ir.CallExpr)
		if !ok {
			return
		}
		switch op := ir.LogosRegionCallOp(call); op {
		case ir.LogosRegionOpen:
			if ir.LogosRegions.IsLive(call) {
				p.seq[call] = order
				order++
				p.relevant = true
			}
		case ir.LogosRegionNone:
		default:
			if ir.LogosRegions.Placed(call) {
				if r := p.regionName(call); r != nil {
					p.sites[call] = r
				}
				p.relevant = true
			}
		}
	})
	if !p.relevant {
		return p
	}
	for call, ow := range ir.LogosRegionOwners(fn) {
		if ir.LogosRegions.IsLive(call) {
			k := logosRegionOwnerKey{ow.Stmt, ow.List}
			p.live[k] = append(p.live[k], call)
		}
	}
	return p
}

func (p *logosRegionLowering) ownerFor(k logosRegionOwnerKey, parent *logosRegionOwner, loop, caseOwner, clause bool) *logosRegionOwner {
	o := &logosRegionOwner{stmt: k.stmt, parent: parent, loop: loop, caseOwner: caseOwner, clause: clause}
	o.closables = p.live[k]
	p.sortClosables(o.closables)
	return o
}

func (p *logosRegionLowering) funcOwner(list *ir.Nodes) *logosRegionOwner {
	return p.ownerFor(logosRegionOwnerKey{nil, list}, nil, false, false, false)
}

func (p *logosRegionLowering) listOwner(stmt ir.Node, list *ir.Nodes, parent *logosRegionOwner) *logosRegionOwner {
	return p.ownerFor(logosRegionOwnerKey{stmt, list}, parent, false, false, false)
}

func (p *logosRegionLowering) loopOwner(stmt ir.Node, list *ir.Nodes, parent *logosRegionOwner) *logosRegionOwner {
	return p.ownerFor(logosRegionOwnerKey{stmt, list}, parent, true, false, false)
}

func (p *logosRegionLowering) caseOwnerFor(stmt ir.Node, list *ir.Nodes, parent *logosRegionOwner, extra []*ir.CallExpr) *logosRegionOwner {
	o := p.ownerFor(logosRegionOwnerKey{stmt, list}, parent, false, true, false)
	o.closables = append(o.closables, extra...)
	p.sortClosables(o.closables)
	return o
}

func (p *logosRegionLowering) clauseOwner(stmt ir.Node, parent *logosRegionOwner) *logosRegionOwner {
	return p.ownerFor(logosRegionOwnerKey{stmt, nil}, parent, false, false, true)
}

func (p *logosRegionLowering) sortClosables(calls []*ir.CallExpr) {
	slices.SortStableFunc(calls, func(a, b *ir.CallExpr) int {
		return p.seq[a] - p.seq[b]
	})
}

func (p *logosRegionLowering) lowerBlockList(list *ir.Nodes, stmt ir.Node, parent *logosRegionOwner) {
	p.lowerOwned(list, p.listOwner(stmt, list, parent))
}

func (p *logosRegionLowering) lowerLoopBody(list *ir.Nodes, stmt ir.Node, parent *logosRegionOwner) {
	p.lowerOwned(list, p.loopOwner(stmt, list, parent))
}

func (p *logosRegionLowering) lowerCaseList(list *ir.Nodes, stmt ir.Node, parent *logosRegionOwner, extra []*ir.CallExpr) {
	p.lowerOwned(list, p.caseOwnerFor(stmt, list, parent, extra))
}

func (p *logosRegionLowering) lowerClauseList(list *ir.Nodes, stmt ir.Node, parent *logosRegionOwner) *logosRegionOwner {
	own := p.clauseOwner(stmt, parent)
	p.lowerOwned(list, own)
	return own
}

func (p *logosRegionLowering) lowerOwned(list *ir.Nodes, own *logosRegionOwner) {
	if p.labels == nil {
		p.labels = make(map[*types.Sym]ir.Node)
	}
	var pending *types.Sym
	for i := 0; i < len(*list); i++ {
		st := (*list)[i]
		if lb, ok := st.(*ir.LabelStmt); ok {
			pending = lb.Label
			continue
		}
		if pending != nil {
			p.labels[pending] = st
			pending = nil
		}
		switch st.Op() {
		case ir.ORETURN:
			i = p.lowerReturn(list, i, st.(*ir.ReturnStmt), own)
		case ir.OBREAK, ir.OCONTINUE, ir.OFALL:
			if closes := p.jumpCloses(st.(*ir.BranchStmt), own); len(closes) > 0 {
				pspliceAt(list, i, closes)
				i += len(closes)
			}
		case ir.OBLOCK:
			blk := st.(*ir.BlockStmt)
			p.lowerBlockList(&blk.List, blk, own)
		case ir.OFOR:
			i = p.lowerFor(list, i, st.(*ir.ForStmt), own)
		case ir.ORANGE:
			p.lowerRange(st.(*ir.RangeStmt), own)
		case ir.OIF:
			i = p.lowerIf(list, i, st.(*ir.IfStmt), own)
		case ir.OSWITCH:
			i = p.lowerSwitch(list, i, st.(*ir.SwitchStmt), own)
		case ir.OSELECT:
			p.lowerSelect(st.(*ir.SelectStmt), own)
		case ir.OAS:
			p.lowerOpenStmt(st.(*ir.AssignStmt))
			p.lowerPlain(st)
		case ir.OCALLFUNC:
			if call, ok := st.(*ir.CallExpr); ok && ir.LogosRegionCallOp(call) == ir.LogosRegionOpen {
				if rep := p.lowerOpenBare(call); rep != nil {
					(*list)[i] = rep
					st = rep
				}
			}
			p.lowerPlain(st)
		default:
			p.lowerPlain(st)
		}
	}
	if !own.clause && !logosRegionListTerminates(*list) {
		list.Append(own.ownCloses(p)...)
	}
}

func (p *logosRegionLowering) lowerFor(list *ir.Nodes, i int, f *ir.ForStmt, own *logosRegionOwner) int {
	parent := own
	var clauses *logosRegionOwner
	if len(f.Init()) > 0 {
		clauses = p.lowerClauseList(f.PtrInit(), f, own)
		parent = clauses
	}
	p.lowerLoopBody(&f.Body, f, parent)
	p.withInit(f.PtrInit(), func() {
		p.descend(f.Cond)
		p.descend(f.Post)
	})
	if clauses != nil {
		if tail := clauses.ownCloses(p); len(tail) > 0 {
			pspliceAfter(list, i, tail)
			i += len(tail)
		}
	}
	return i
}

func (p *logosRegionLowering) lowerIf(list *ir.Nodes, i int, n *ir.IfStmt, own *logosRegionOwner) int {
	parent := own
	var clauses *logosRegionOwner
	if len(n.Init()) > 0 {
		clauses = p.lowerClauseList(n.PtrInit(), n, own)
		parent = clauses
	}
	p.lowerBlockList(&n.Body, n, parent)
	if len(n.Else) > 0 {
		p.lowerBlockList(&n.Else, n, parent)
	}
	p.withInit(n.PtrInit(), func() {
		p.descend(n.Cond)
	})
	if clauses != nil {
		if tail := clauses.ownCloses(p); len(tail) > 0 {
			pspliceAfter(list, i, tail)
			i += len(tail)
		}
	}
	return i
}

func (p *logosRegionLowering) lowerSwitch(list *ir.Nodes, i int, sw *ir.SwitchStmt, own *logosRegionOwner) int {
	parent := own
	var clauses *logosRegionOwner
	if len(sw.Init()) > 0 {
		clauses = p.lowerClauseList(sw.PtrInit(), sw, own)
		parent = clauses
	}
	for _, c := range sw.Cases {
		if c == nil {
			continue
		}
		p.lowerCaseList(&c.Body, c, parent, nil)
		p.withInit(nil, func() {
			for _, e := range c.List {
				p.descend(e)
			}
		})
	}
	p.withInit(sw.PtrInit(), func() {
		p.descend(sw.Tag)
	})
	if clauses != nil {
		if tail := clauses.ownCloses(p); len(tail) > 0 {
			pspliceAfter(list, i, tail)
			i += len(tail)
		}
	}
	return i
}

func (p *logosRegionLowering) lowerSelect(sel *ir.SelectStmt, own *logosRegionOwner) {
	for _, cc := range sel.Cases {
		if cc == nil {
			continue
		}
		var comm []*ir.CallExpr
		if cc.Comm != nil {
			comm = p.live[logosRegionOwnerKey{cc.Comm, nil}]
			p.withInit(nil, func() {
				p.descend(cc.Comm)
			})
		}
		p.lowerCaseList(&cc.Body, cc, own, comm)
	}
}

func (p *logosRegionLowering) lowerRange(rg *ir.RangeStmt, own *logosRegionOwner) {
	p.lowerLoopBody(&rg.Body, rg, own)
	p.withInit(rg.PtrInit(), func() {
		p.descend(rg.X)
	})
}

func (p *logosRegionLowering) lowerPlain(st ir.Node) {
	var init *ir.Nodes
	if ps, ok := st.(interface{ PtrInit() *ir.Nodes }); ok {
		init = ps.PtrInit()
	}
	p.withInit(init, func() {
		p.descend(st)
	})
}

func (p *logosRegionLowering) withInit(init *ir.Nodes, f func()) {
	saved := p.init
	p.init = init
	f()
	p.init = saved
}

func (p *logosRegionLowering) descend(n ir.Node) {
	if n == nil {
		return
	}
	ir.EditChildren(n, func(c ir.Node) ir.Node {
		if cl, ok := c.(*ir.ClosureExpr); ok {
			if cl.Func != nil {
				p.lowerOwned(&cl.Func.Body, p.funcOwner(&cl.Func.Body))
			}
			return c
		}
		if call, ok := c.(*ir.CallExpr); ok {
			switch op := ir.LogosRegionCallOp(call); op {
			case ir.LogosRegionOpen, ir.LogosRegionNone:
			default:
				return p.lowerSite(call, op)
			}
		}
		p.descend(c)
		return c
	})
}

func (p *logosRegionLowering) lowerReturn(list *ir.Nodes, i int, ret *ir.ReturnStmt, own *logosRegionOwner) int {
	var pre ir.Nodes
	results := []ir.Node(ret.Results)
	for _, e := range results {
		if e == nil {
			continue
		}
		t := typecheck.TempAt(ret.Pos(), p.fn, e.Type())
		oas := ir.NewAssignStmt(ret.Pos(), t, e)
		oas.SetTypecheck(1)
		pre = append(pre, oas)
	}
	if len(pre) > 0 {
		newResults := make(ir.Nodes, len(pre))
		for k := range pre {
			newResults[k] = pre[k].(*ir.AssignStmt).X
		}
		ret.Results = newResults
	}
	tail := append(pre, p.chainCloses(own)...)
	pspliceAt(list, i, tail)
	for _, s := range pre {
		p.lowerPlain(s)
	}
	return i + len(tail)
}

func (p *logosRegionLowering) jumpCloses(b *ir.BranchStmt, own *logosRegionOwner) []ir.Node {
	switch {
	case b.Op() == ir.OFALL:
		return own.ownCloses(p)
	case b.Label != nil:
		return p.labeledJumpCloses(b, own)
	case b.Op() == ir.OCONTINUE:
		return p.loopExitCloses(own, false)
	case b.Op() == ir.OBREAK:
		return p.loopExitCloses(own, true)
	}
	return nil
}

func (p *logosRegionLowering) labeledJumpCloses(b *ir.BranchStmt, own *logosRegionOwner) []ir.Node {
	target, ok := p.labels[b.Label]
	if !ok {
		return nil
	}
	var stop *logosRegionOwner
	for o := own; o != nil; o = o.parent {
		if o.stmt == target {
			stop = o
		}
	}
	if stop == nil {
		return nil
	}
	var out []ir.Node
	for o := own; ; o = o.parent {
		out = append(out, o.ownCloses(p)...)
		if o == stop {
			return out
		}
	}
}

func (p *logosRegionLowering) loopExitCloses(own *logosRegionOwner, mayBindCase bool) []ir.Node {
	var out []ir.Node
	for o := own; o != nil; o = o.parent {
		out = append(out, o.ownCloses(p)...)
		if o.loop || (mayBindCase && o.caseOwner) {
			return out
		}
	}
	return out
}

func (p *logosRegionLowering) chainCloses(own *logosRegionOwner) []ir.Node {
	var out []ir.Node
	for o := own; o != nil; o = o.parent {
		out = append(out, o.ownCloses(p)...)
	}
	return out
}

func (o *logosRegionOwner) ownCloses(p *logosRegionLowering) []ir.Node {
	if o == nil || len(o.closables) == 0 {
		return nil
	}
	out := make([]ir.Node, 0, len(o.closables))
	for i := len(o.closables) - 1; i >= 0; i-- {
		call := o.closables[i]
		name := p.names[call]
		if name == nil {
			continue
		}
		out = append(out, p.closeCall(name, call.Pos()))
	}
	return out
}

func (p *logosRegionLowering) closeCall(name *ir.Name, pos src.XPos) ir.Node {
	h := p.regionDot(name, 0, pos)
	g := p.regionDot(name, 1, pos)
	old := base.Pos
	base.Pos = pos
	call := typecheck.Call(pos, typecheck.LookupRuntime("logosRegionClose"), []ir.Node{h, g}, false).(*ir.CallExpr)
	base.Pos = old
	call.SetPos(pos)
	return call
}

func (p *logosRegionLowering) regionDot(name *ir.Name, idx int, pos src.XPos) ir.Node {
	f := name.Type().Field(idx)
	d := ir.NewSelectorExpr(pos, ir.ODOT, name, f.Sym)
	d.Selection = f
	d.SetType(f.Type)
	d.SetTypecheck(1)
	return d
}

func (p *logosRegionLowering) lowerOpenStmt(as *ir.AssignStmt) {
	call, ok := as.Y.(*ir.CallExpr)
	if !ok || ir.LogosRegionCallOp(call) != ir.LogosRegionOpen {
		return
	}
	if !ir.LogosRegions.IsLive(call) {
		lit := ir.NewCompLitExpr(call.Pos(), ir.OSTRUCTLIT, call.Type(), nil)
		lit.SetTypecheck(1)
		as.Y = lit
		return
	}
	if p.names[call] == nil {
		return
	}
	as.Y = p.openLit(call, as.PtrInit())
}

func (p *logosRegionLowering) lowerOpenBare(call *ir.CallExpr) ir.Node {
	if !ir.LogosRegions.IsLive(call) {
		return nil
	}
	tmp := typecheck.TempAt(call.Pos(), p.fn, call.Type())
	var init ir.Nodes
	oas := ir.NewAssignStmt(call.Pos(), tmp, p.openLit(call, &init))
	oas.SetTypecheck(1)
	init.Append(oas)
	p.names[call] = tmp
	return ir.NewBlockStmt(call.Pos(), init)
}
func (p *logosRegionLowering) openLit(call *ir.CallExpr, init *ir.Nodes) ir.Node {
	pos := call.Pos()
	tmpH := typecheck.TempAt(pos, p.fn, types.Types[types.TUNSAFEPTR])
	tmpG := typecheck.TempAt(pos, p.fn, types.Types[types.TUINTPTR])
	var callInit ir.Nodes
	openFn := typecheck.LookupRuntime("logosRegionOpen")
	old := base.Pos
	base.Pos = pos
	openCall := mkcall1(openFn, openFn.Type().ResultsTuple(), &callInit)
	base.Pos = old
	openCall.SetPos(pos)
	init.Append(callInit...)
	oas2 := ir.NewAssignListStmt(pos, ir.OAS2FUNC, []ir.Node{tmpH, tmpG}, []ir.Node{openCall})
	oas2.SetTypecheck(1)
	init.Append(oas2)
	t := call.Type()
	lit := ir.NewCompLitExpr(pos, ir.OSTRUCTLIT, t, []ir.Node{
		p.fieldExpr(t, 0, tmpH, pos),
		p.fieldExpr(t, 1, tmpG, pos),
	})
	lit.SetTypecheck(1)
	return lit
}

func (p *logosRegionLowering) fieldExpr(t *types.Type, idx int, v ir.Node, pos src.XPos) ir.Node {
	f := t.Field(idx)
	e := ir.NewStructKeyExpr(pos, f, v)
	e.SetType(f.Type)
	e.SetTypecheck(1)
	return e
}

func (p *logosRegionLowering) lowerSite(call *ir.CallExpr, op ir.LogosRegionOp) ir.Node {
	if !ir.LogosRegions.Placed(call) || len(call.Args) == 0 {
		return call
	}
	r := p.sites[call]
	if r == nil {
		r = p.regionName(call)
	}
	if r == nil {
		return call
	}
	pos := call.Pos()
	h := p.regionDot(r, 0, pos)
	g := p.regionDot(r, 1, pos)
	switch op {
	case ir.LogosRegionNew:
		return p.lowerSiteNew(call, h, g)
	case ir.LogosRegionMake:
		return p.lowerSiteMake(call, h, g)
	case ir.LogosRegionAppend:
		return p.lowerSiteAppend(call, h, g)
	}
	return call
}
func (p *logosRegionLowering) lowerSiteNew(call *ir.CallExpr, h, g ir.Node) ir.Node {
	elem := ir.LogosRegionElem(call)
	if elem == nil {
		return call
	}
	types.CalcSize(elem)
	size := elem.Size()
	pos := call.Pos()
	init := p.init
	if logosRegionBumpable(elem, size) {
		return typecheck.Conv(p.bumpPointer(pos, elem, size, h, g), call.Type())
	}
	old := base.Pos
	base.Pos = pos
	var out ir.Node
	if elem.HasPointers() {
		fn := typecheck.LookupRuntime("logosRegionAllocScan")
		made := mkcall1(fn, types.Types[types.TUNSAFEPTR], init, h, g, reflectdata.TypePtrAt(pos, elem), ir.NewInt(pos, size))
		made.SetPos(pos)
		out = typecheck.Conv(made, call.Type())
	} else {
		fn := typecheck.LookupRuntime("logosRegionAllocNoscan")
		align := ir.NewInt(pos, elem.Alignment())
		made := mkcall1(fn, types.Types[types.TUNSAFEPTR], init, h, g, ir.NewInt(pos, size), align)
		made.SetPos(pos)
		out = typecheck.Conv(made, call.Type())
	}
	base.Pos = old
	return out
}

func logosRegionBumpable(elem *types.Type, size int64) bool {
	return !elem.HasPointers() && size > 0 && size <= ssagen.LogosArenaBumpMaxBytes && buildcfg.GOARCH == "amd64" && !base.Flag.Race
}

func logosRegionMakeBumpSize(elem *types.Type, length ir.Node) (int64, bool) {
	if !ir.IsConst(length, constant.Int) {
		return 0, false
	}
	count, width := ir.Int64Val(length), elem.Size()
	if count <= 0 || width <= 0 || count > ssagen.LogosArenaBumpMaxBytes/width {
		return 0, false
	}
	size := count * width
	return size, logosRegionBumpable(elem, size)
}

func (p *logosRegionLowering) bumpPointer(pos src.XPos, elem *types.Type, size int64, h, g ir.Node) *ir.Name {
	init := p.init
	pt := typecheck.TempAt(pos, p.fn, types.Types[types.TUNSAFEPTR])
	old := base.Pos
	base.Pos = pos
	callBump := mkcall1(typecheck.LookupRuntime("logosRegionBump"), types.Types[types.TUNSAFEPTR], init, h, g, ir.NewInt(pos, size))
	callBump.SetPos(pos)
	p.appendInit(init, ir.NewAssignStmt(pos, pt, callBump))
	var slow ir.Nodes
	callAlloc := mkcall1(typecheck.LookupRuntime("logosRegionAllocNoscan"), types.Types[types.TUNSAFEPTR], &slow, h, g, ir.NewInt(pos, size), ir.NewInt(pos, elem.Alignment()))
	callAlloc.SetPos(pos)
	p.appendInit(&slow, ir.NewAssignStmt(pos, pt, callAlloc))
	var clr ir.Nodes
	clrCall := mkcall("memclrNoHeapPointers", nil, &clr, pt, ir.NewInt(pos, size))
	clrCall.SetPos(pos)
	p.appendInit(&clr, clrCall)
	base.Pos = old
	nif := ir.NewIfStmt(pos, ir.NewBinaryExpr(pos, ir.OEQ, pt, typecheck.NodNil()), slow, clr)
	p.appendInit(init, nif)
	return pt
}

func (p *logosRegionLowering) makeSliceData(pos src.XPos, elem *types.Type, length, h, g ir.Node) *ir.Name {
	init := p.init
	entry := "logosRegionMakeSliceNoscan"
	if elem.HasPointers() {
		entry = "logosRegionMakeSliceScan"
	}
	pt := typecheck.TempAt(pos, p.fn, types.Types[types.TUNSAFEPTR])
	len64 := typecheck.Conv(length, types.Types[types.TINT64])
	old := base.Pos
	base.Pos = pos
	callMake := mkcall1(typecheck.LookupRuntime(entry), types.Types[types.TUNSAFEPTR], init, h, g, reflectdata.TypePtrAt(pos, elem), len64, len64)
	callMake.SetPos(pos)
	base.Pos = old
	p.appendInit(init, ir.NewAssignStmt(pos, pt, callMake))
	return pt
}

func (p *logosRegionLowering) lowerSiteMake(call *ir.CallExpr, h, g ir.Node) ir.Node {
	argBase := ir.LogosRegionArgBase(call)
	if len(call.Args) != argBase+2 {
		return call
	}
	pos := call.Pos()
	init := p.init
	elem := call.Type().Elem()
	types.CalcSize(elem)
	length := safeExpr(call.Args[argBase+1], init)
	var data ir.Node
	if size, ok := logosRegionMakeBumpSize(elem, length); ok {
		data = p.bumpPointer(pos, elem, size, h, g)
	} else {
		data = p.makeSliceData(pos, elem, length, h, g)
	}
	slice := typecheck.TempAt(pos, p.fn, call.Type())
	n := typecheck.Conv(length, types.Types[types.TINT])
	sh := ir.NewSliceHeaderExpr(pos, call.Type(), data, n, n)
	sh.SetTypecheck(1)
	p.appendInit(init, ir.NewAssignStmt(pos, slice, sh))
	return slice
}

func (p *logosRegionLowering) lowerSiteAppend(call *ir.CallExpr, h, g ir.Node) ir.Node {
	argBase := ir.LogosRegionArgBase(call)
	if len(call.Args) < argBase+2 {
		return call
	}
	app := ir.NewCallExpr(call.Pos(), ir.OAPPEND, nil, []ir.Node(call.Args[argBase+1:]))
	app.IsDDD = call.IsDDD
	app.SetType(call.Type())
	app.SetTypecheck(1)
	p.appends = append(p.appends, logosRegionAppendEntry{app, h, g})
	return app
}

func (p *logosRegionLowering) appendInit(init *ir.Nodes, stmt ir.Node) {
	appendWalkStmt(init, stmt)
}
func (p *logosRegionLowering) regionName(call *ir.CallExpr) *ir.Name {
	argBase := ir.LogosRegionArgBase(call)
	if len(call.Args) <= argBase {
		return nil
	}
	r := call.Args[argBase]
	for range 16 {
		name, ok := r.(*ir.Name)
		if !ok {
			return nil
		}
		switch name.Class {
		case ir.PPARAM:
			return name
		case ir.PAUTO, ir.PAUTOHEAP:
		default:
			return nil
		}
		defn, ok := name.Defn.(*ir.AssignStmt)
		if !ok || defn.Y == nil {
			return nil
		}
		switch y := defn.Y.(type) {
		case *ir.CallExpr:
			if ir.LogosRegionCallOp(y) == ir.LogosRegionOpen {
				return name
			}
			return nil
		case *ir.Name:
			r = y
		default:
			return nil
		}
	}
	return nil
}

type logosRegionGrow struct {
	app    *ir.CallExpr
	slice  *ir.Name
	init   *ir.Nodes
	oldPtr ir.Node
	newLen ir.Node
	oldCap ir.Node
	num    ir.Node
}

func logosRegionGrowslice(g logosRegionGrow) *ir.CallExpr {
	if v, ok := ssagen.LogosRegionAppends.Load(g.app); ok {
		info := v.(ssagen.LogosRegionGrowInfo)
		elem := g.slice.Type().Elem()
		fn := typecheck.LookupRuntime(ssagen.LogosRegionGrowSliceEntry(elem), elem, elem)
		pos := g.app.Pos()
		old := base.Pos
		base.Pos = pos
		made := mkcall1(fn, g.slice.Type(), g.init, info.H, info.G, g.oldPtr, g.newLen, g.oldCap, g.num, reflectdata.TypePtrAt(pos, elem))
		base.Pos = old
		made.SetPos(pos)
		return made
	}
	return walkGrowslice(g.slice, g.init, g.oldPtr, g.newLen, g.oldCap, g.num)
}

func pspliceAt(list *ir.Nodes, i int, ins []ir.Node) {
	old := *list
	newList := make(ir.Nodes, 0, len(old)+len(ins))
	newList = append(newList, old[:i]...)
	newList = append(newList, ins...)
	newList = append(newList, old[i:]...)
	*list = newList
}

func pspliceAfter(list *ir.Nodes, i int, tail []ir.Node) {
	old := *list
	newList := make(ir.Nodes, 0, len(old)+len(tail))
	newList = append(newList, old[:i+1]...)
	newList = append(newList, tail...)
	newList = append(newList, old[i+1:]...)
	*list = newList
}

func logosRegionListTerminates(list []ir.Node) bool {
	if len(list) == 0 {
		return false
	}
	return logosRegionTerminates(list[len(list)-1])
}

func logosRegionTerminates(st ir.Node) bool {
	switch st.Op() {
	case ir.ORETURN, ir.OBREAK, ir.OCONTINUE, ir.OGOTO, ir.OFALL, ir.OPANIC, ir.OTAILCALL:
		return true
	case ir.OBLOCK:
		return logosRegionListTerminates(st.(*ir.BlockStmt).List)
	case ir.OIF:
		i := st.(*ir.IfStmt)
		return len(i.Else) > 0 && logosRegionListTerminates(i.Else) && logosRegionListTerminates(i.Body)
	case ir.OSWITCH:
		return logosRegionSwitchTerminates(st.(*ir.SwitchStmt))
	}
	return false
}

func logosRegionSwitchTerminates(sw *ir.SwitchStmt) bool {
	hasDefault := false
	for _, c := range sw.Cases {
		if c == nil {
			continue
		}
		if len(c.List) == 0 {
			hasDefault = true
		}
		if !logosRegionListTerminates(c.Body) {
			return false
		}
	}
	return hasDefault
}
