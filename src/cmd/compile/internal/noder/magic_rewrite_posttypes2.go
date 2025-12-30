package noder

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// rewriteMagicPostTypes2 performs a second-stage rewrite after the second types2
// pass. It enables cross-package syntax sugar by lowering:
// - binary operators (a+b, a==b, ...) to magic method calls (_add/_eq/...)
// - index/slice expressions (a[i], a[i:j], a[i,j], ...) to _getitem
// - index assignments (a[i]=v, a[i,j]=v, ...) to _setitem (value-first)
// - constructor make(T, ...) to (&T{})._init(...)
//
// This pass is intentionally placed after types2 so we can use full type
// information across packages to pick the correct overload-suffixed magic method.
func rewriteMagicPostTypes2(noders []*noder, pkg *types2.Package, info *types2.Info) {
	if len(noders) == 0 || pkg == nil || info == nil {
		return
	}
	r := &postTypes2MagicRewriter{pkg: pkg, info: info}
	for _, n := range noders {
		if n == nil || n.file == nil {
			continue
		}
		r.rewriteFile(n.file)
	}
}

type postTypes2MagicRewriter struct {
	pkg  *types2.Package
	info *types2.Info
}

func (r *postTypes2MagicRewriter) rewriteFile(f *syntax.File) {
	for i, d := range f.DeclList {
		f.DeclList[i] = r.rewriteDecl(d)
	}
}

func (r *postTypes2MagicRewriter) rewriteDecl(d syntax.Decl) syntax.Decl {
	switch d := d.(type) {
	case *syntax.FuncDecl:
		if d.Body != nil {
			r.rewriteBlock(d.Body)
		}
		return d
	default:
		return d
	}
}

func (r *postTypes2MagicRewriter) rewriteBlock(b *syntax.BlockStmt) {
	if b == nil {
		return
	}
	for i, st := range b.List {
		b.List[i] = r.rewriteStmt(st)
	}
}

func (r *postTypes2MagicRewriter) rewriteStmt(st syntax.Stmt) syntax.Stmt {
	if st == nil {
		return nil
	}
	switch s := st.(type) {
	case *syntax.BlockStmt:
		r.rewriteBlock(s)
		return s
	case *syntax.ExprStmt:
		s.X = r.rewriteExpr(s.X)
		return s
	case *syntax.SendStmt:
		// Lower "a <- v" to a._send(v) for non-channel receivers.
		// Keep native channel send unchanged.
		s.Chan = r.rewriteExpr(s.Chan)
		s.Value = r.rewriteExpr(s.Value)
		chT := r.typeOf(s.Chan)
		valT := r.typeOf(s.Value)
		if chT != nil && valT != nil {
			if _, ok := types2.CoreType(chT).(*types2.Chan); !ok {
				if name, _, ok := r.lookupBestMagic(chT, "_send", []types2.Type{valT}, 0, false); ok {
					return &syntax.ExprStmt{X: r.makeCall(s.Pos(), s.Chan, name, []syntax.Expr{s.Value})}
				}
			}
		}
		return s
	case *syntax.AssignStmt:
		// Handle a[idx] = v lowering to _setitem(value, idx...)
		if s.Op == 0 && s.Rhs != nil {
			if call := r.tryRewriteIndexAssignToSetItem(s.Pos(), s.Lhs, s.Rhs); call != nil {
				return &syntax.ExprStmt{X: call}
			}
		}
		s.Lhs = r.rewriteExpr(s.Lhs)
		if s.Rhs != nil {
			s.Rhs = r.rewriteExpr(s.Rhs)
		}
		return s
	case *syntax.IfStmt:
		if s.Init != nil {
			s.Init = r.rewriteSimpleStmt(s.Init)
		}
		s.Cond = r.rewriteExpr(s.Cond)
		r.rewriteBlock(s.Then)
		if s.Else != nil {
			s.Else = r.rewriteStmt(s.Else)
		}
		return s
	case *syntax.ForStmt:
		if s.Init != nil {
			s.Init = r.rewriteSimpleStmt(s.Init)
		}
		if s.Cond != nil {
			s.Cond = r.rewriteExpr(s.Cond)
		}
		if s.Post != nil {
			s.Post = r.rewriteSimpleStmt(s.Post)
		}
		r.rewriteBlock(s.Body)
		return s
	case *syntax.SwitchStmt:
		if s.Init != nil {
			s.Init = r.rewriteSimpleStmt(s.Init)
		}
		if s.Tag != nil {
			s.Tag = r.rewriteExpr(s.Tag)
		}
		for i := range s.Body {
			cc := s.Body[i]
			if cc == nil {
				continue
			}
			if cc.Cases != nil {
				cc.Cases = r.rewriteExpr(cc.Cases)
			}
			for j := range cc.Body {
				cc.Body[j] = r.rewriteStmt(cc.Body[j])
			}
		}
		return s
	case *syntax.ReturnStmt:
		if s.Results != nil {
			s.Results = r.rewriteExpr(s.Results)
		}
		return s
	case *syntax.DeclStmt:
		// Best-effort: rewrite expressions inside init decls.
		for i, d := range s.DeclList {
			s.DeclList[i] = r.rewriteDecl(d)
		}
		return s
	case *syntax.LabeledStmt:
		s.Stmt = r.rewriteStmt(s.Stmt)
		return s
	case *syntax.BranchStmt:
		return s
	case *syntax.CallStmt:
		if call, _ := r.rewriteExpr(s.Call).(*syntax.CallExpr); call != nil {
			s.Call = call
		}
		if s.DeferAt != nil {
			s.DeferAt = r.rewriteExpr(s.DeferAt)
		}
		return s
	default:
		// Fall back to rewriting child expressions we care about.
		return s
	}
}

func (r *postTypes2MagicRewriter) rewriteSimpleStmt(st syntax.SimpleStmt) syntax.SimpleStmt {
	if st == nil {
		return nil
	}
	switch s := st.(type) {
	case *syntax.AssignStmt:
		// rewriteStmt may lower it into an ExprStmt (index assignment -> _setitem call).
		if out := r.rewriteStmt(s); out != nil {
			if as, _ := out.(*syntax.AssignStmt); as != nil {
				return as
			}
			if es, _ := out.(*syntax.ExprStmt); es != nil {
				return es
			}
		}
		return s
	case *syntax.ExprStmt:
		return r.rewriteStmt(s).(*syntax.ExprStmt)
	case *syntax.SendStmt:
		// rewriteStmt may lower it into an ExprStmt (magic _send call).
		if out := r.rewriteStmt(s); out != nil {
			if ss, _ := out.(*syntax.SendStmt); ss != nil {
				return ss
			}
			if es, _ := out.(*syntax.ExprStmt); es != nil {
				return es
			}
		}
		return s
	default:
		return s
	}
}

func (r *postTypes2MagicRewriter) rewriteExpr(e syntax.Expr) syntax.Expr {
	if e == nil {
		return nil
	}
	switch x := e.(type) {
	case *syntax.ParenExpr:
		x.X = r.rewriteExpr(x.X)
		return x
	case *syntax.Operation:
		x.X = r.rewriteExpr(x.X)
		if x.Y != nil {
			x.Y = r.rewriteExpr(x.Y)
		}
		if call := r.tryRewriteOperationToMagicCall(x); call != nil {
			return call
		}
		return x
	case *syntax.IndexExpr:
		x.X = r.rewriteExpr(x.X)
		x.Index = r.rewriteExpr(x.Index)
		if call := r.tryRewriteIndexToGetItem(x); call != nil {
			return call
		}
		return x
	case *syntax.SliceExpr:
		// X==nil slice-exprs are index descriptors; keep them unchanged.
		if x.X != nil {
			x.X = r.rewriteExpr(x.X)
		}
		for i := range x.Index {
			if x.Index[i] != nil {
				x.Index[i] = r.rewriteExpr(x.Index[i])
			}
		}
		if call := r.tryRewriteSliceToGetItem(x); call != nil {
			return call
		}
		return x
	case *syntax.CallExpr:
		x.Fun = r.rewriteExpr(x.Fun)
		for i := range x.ArgList {
			x.ArgList[i] = r.rewriteExpr(x.ArgList[i])
		}
		if call := r.tryRewriteMakeToInit(x); call != nil {
			return call
		}
		return x
	case *syntax.SelectorExpr:
		x.X = r.rewriteExpr(x.X)
		return x
	case *syntax.ListExpr:
		for i := range x.ElemList {
			x.ElemList[i] = r.rewriteExpr(x.ElemList[i])
		}
		return x
	case *syntax.CompositeLit:
		if x.Type != nil {
			x.Type = r.rewriteExpr(x.Type)
		}
		for i := range x.ElemList {
			x.ElemList[i] = r.rewriteExpr(x.ElemList[i])
		}
		return x
	default:
		return e
	}
}

// ---- helpers: type info ----

func (r *postTypes2MagicRewriter) typeAndValue(e syntax.Expr) syntax.TypeAndValue {
	if e == nil {
		return syntax.TypeAndValue{}
	}
	tv := e.GetTypeInfo()
	// Mirror pkgWriter.maybeTypeAndValue: if this is an identifier with an instantiated
	// generic function/type, use the instantiated type for downstream logic.
	if r.info != nil {
		if name, ok := e.(*syntax.Name); ok {
			if inst, ok := r.info.Instances[name]; ok {
				tv.Type = inst.Type
			}
			// Fallback: Names often have their type available via Uses/Defs even if
			// TypeAndValue wasn't populated on the node (Info.Types may be nil).
			if tv.Type == nil {
				if obj := r.info.Uses[name]; obj != nil {
					tv.Type = obj.Type()
					tv.SetIsValue()
				} else if obj := r.info.Defs[name]; obj != nil {
					tv.Type = obj.Type()
					tv.SetIsValue()
				}
			}
		}
	}
	return tv
}

func (r *postTypes2MagicRewriter) typeOf(e syntax.Expr) types2.Type {
	tv := r.typeAndValue(e)
	return tv.Type
}

func (r *postTypes2MagicRewriter) isValueExpr(e syntax.Expr) bool {
	return r.typeAndValue(e).IsValue()
}

// ---- magic method selection ----

func isMagicOverloadName(name, base string) bool {
	if name == base {
		return true
	}
	return len(name) > len(base) && name[:len(base)] == base && name[len(base)] == '_'
}

func (r *postTypes2MagicRewriter) collectMagicCandidateNames(recv types2.Type, base string) []string {
	seen := make(map[string]bool)
	var out []string

	add := func(n string) {
		if !isMagicOverloadName(n, base) {
			return
		}
		if !seen[n] {
			seen[n] = true
			out = append(out, n)
		}
	}

	// Named types: declared methods
	if n, _ := types2.Unalias(recv).(*types2.Named); n != nil {
		for i := 0; i < n.NumMethods(); i++ {
			m := n.Method(i)
			if m != nil {
				add(m.Name())
			}
		}
	}
	// Pointer to named: declared methods are on base named
	if p, _ := types2.Unalias(recv).(*types2.Pointer); p != nil {
		if n, _ := types2.Unalias(p.Elem()).(*types2.Named); n != nil {
			for i := 0; i < n.NumMethods(); i++ {
				m := n.Method(i)
				if m != nil {
					add(m.Name())
				}
			}
		}
	}
	// Interfaces (incl. type param constraints): method set
	if it, _ := types2.CoreType(recv).(*types2.Interface); it != nil {
		for i := 0; i < it.NumMethods(); i++ {
			m := it.Method(i)
			if m != nil {
				add(m.Name())
			}
		}
	}
	// Always allow the base name as fallback (important for synthesized basics).
	add(base)
	return out
}

func (r *postTypes2MagicRewriter) matchMagicSignature(sig *types2.Signature, argTypes []types2.Type, wantResults int, wantBoolResult bool) (score int, ok bool) {
	if sig == nil {
		return 0, false
	}
	if sig.Results().Len() != wantResults {
		return 0, false
	}
	if wantBoolResult {
		if wantResults != 1 || !types2.Identical(sig.Results().At(0).Type(), types2.Typ[types2.Bool]) {
			return 0, false
		}
	}

	nargs := len(argTypes)
	npars := sig.Params().Len()
	if !sig.Variadic() {
		if nargs != npars {
			return 0, false
		}
	} else {
		if npars == 0 || nargs < npars-1 {
			return 0, false
		}
	}

	score = 0
	for i := 0; i < nargs; i++ {
		var ptyp types2.Type
		if sig.Variadic() && i >= npars-1 {
			// last param is a slice; match its element type
			if sl, _ := types2.CoreType(sig.Params().At(npars - 1).Type()).(*types2.Slice); sl != nil {
				ptyp = sl.Elem()
			} else {
				ptyp = sig.Params().At(npars - 1).Type()
			}
		} else {
			ptyp = sig.Params().At(i).Type()
		}
		atyp := argTypes[i]
		if atyp == nil || ptyp == nil {
			return 0, false
		}
		if types2.Identical(atyp, ptyp) {
			score += 3
			continue
		}
		if types2.AssignableTo(atyp, ptyp) {
			score += 2
			continue
		}
		return 0, false
	}
	return score, true
}

func (r *postTypes2MagicRewriter) lookupBestMagic(recv types2.Type, base string, argTypes []types2.Type, wantResults int, wantBoolResult bool) (methodName string, sig *types2.Signature, ok bool) {
	bestScore := -1
	var bestName string
	var bestSig *types2.Signature
	for _, name := range r.collectMagicCandidateNames(recv, base) {
		obj, _, _ := types2.LookupFieldOrMethod(recv, false, r.pkg, name)
		fn, _ := obj.(*types2.Func)
		if fn == nil {
			continue
		}
		s, _ := fn.Type().(*types2.Signature)
		if s == nil {
			continue
		}
		if sc, ok := r.matchMagicSignature(s, argTypes, wantResults, wantBoolResult); ok && sc > bestScore {
			bestScore = sc
			bestName = name
			bestSig = s
		}
	}
	if bestName == "" || bestSig == nil {
		return "", nil, false
	}
	return bestName, bestSig, true
}

// ---- operator lowering ----

func opToMagic(op syntax.Operator) string {
	switch op {
	case syntax.Add:
		return "_add"
	case syntax.Sub:
		return "_sub"
	case syntax.Mul:
		return "_mul"
	case syntax.Div:
		return "_div"
	case syntax.Rem:
		return "_mod"
	case syntax.And:
		return "_and"
	case syntax.Or:
		return "_or"
	case syntax.Xor:
		return "_xor"
	case syntax.AndNot:
		return "_bitclear"
	case syntax.Shl:
		return "_lshift"
	case syntax.Shr:
		return "_rshift"
	case syntax.Eql:
		return "_eq"
	case syntax.Neq:
		return "_ne"
	case syntax.Lss:
		return "_lt"
	case syntax.Leq:
		return "_le"
	case syntax.Gtr:
		return "_gt"
	case syntax.Geq:
		return "_ge"
	}
	return ""
}

func opToReverseMagic(op syntax.Operator) string {
	switch op {
	case syntax.Add:
		return "_radd"
	case syntax.Sub:
		return "_rsub"
	case syntax.Mul:
		return "_rmul"
	case syntax.Div:
		return "_rdiv"
	case syntax.Rem:
		return "_rmod"
	case syntax.And:
		return "_rand"
	case syntax.Or:
		return "_ror"
	case syntax.Xor:
		return "_rxor"
	case syntax.AndNot:
		return "_rbitclear"
	case syntax.Shl:
		return "_rlshift"
	case syntax.Shr:
		return "_rrshift"
	// Comparisons: mirror fallback.
	case syntax.Eql:
		return "_eq"
	case syntax.Neq:
		return "_ne"
	case syntax.Lss:
		return "_gt"
	case syntax.Leq:
		return "_ge"
	case syntax.Gtr:
		return "_lt"
	case syntax.Geq:
		return "_le"
	}
	return ""
}

func isComparison(op syntax.Operator) bool {
	switch op {
	case syntax.Eql, syntax.Neq, syntax.Lss, syntax.Leq, syntax.Gtr, syntax.Geq:
		return true
	}
	return false
}

func (r *postTypes2MagicRewriter) tryRewriteOperationToMagicCall(op *syntax.Operation) syntax.Expr {
	if op == nil || op.Y == nil {
		if op == nil {
			return nil
		}
		// Do not rewrite constant unary expressions (e.g. -1, ^0, +3).
		// Rewriting them into method calls would change constant-ness and can
		// perturb defaulting behavior in the standard library.
		if tv := r.typeAndValue(op.X); tv.Value != nil {
			return nil
		}
		xt := r.typeOf(op.X)
		if xt == nil {
			return nil
		}
		// Never rewrite unary ops on basic (including untyped) values.
		// This preserves native constant/default-typing behavior in stdlib.
		if _, ok := types2.CoreType(xt).(*types2.Basic); ok {
			return nil
		}
		switch op.Op {
		case syntax.Add:
			if name, _, ok := r.lookupBestMagic(xt, "_pos", nil, 1, false); ok {
				return r.makeCall(op.Pos(), op.X, name, nil)
			}
		case syntax.Sub:
			if name, _, ok := r.lookupBestMagic(xt, "_neg", nil, 1, false); ok {
				return r.makeCall(op.Pos(), op.X, name, nil)
			}
		case syntax.Xor:
			if name, _, ok := r.lookupBestMagic(xt, "_invert", nil, 1, false); ok {
				return r.makeCall(op.Pos(), op.X, name, nil)
			}
		case syntax.Recv:
			// Only rewrite "<-ch" to ch._recv() for non-channel receivers.
			// Keep native channel receive unchanged.
			if _, ok := types2.CoreType(xt).(*types2.Chan); !ok {
				if name, _, ok := r.lookupBestMagic(xt, "_recv", nil, 1, false); ok {
					return r.makeCall(op.Pos(), op.X, name, nil)
				}
			}
		}
		return nil
	}
	base := opToMagic(op.Op)
	if base == "" {
		return nil
	}
	xt := r.typeOf(op.X)
	yt := r.typeOf(op.Y)
	if xt == nil || yt == nil {
		return nil
	}
	wantBool := isComparison(op.Op)

	// Forward: x._add(y)
	if _, ok := types2.CoreType(xt).(*types2.Basic); !ok {
		if name, _, ok := r.lookupBestMagic(xt, base, []types2.Type{yt}, 1, wantBool); ok {
			return r.makeCall(op.Pos(), op.X, name, []syntax.Expr{op.Y})
		}
	}
	// Reverse: y._radd(x)
	if rbase := opToReverseMagic(op.Op); rbase != "" {
		if _, ok := types2.CoreType(yt).(*types2.Basic); !ok {
			if name, _, ok := r.lookupBestMagic(yt, rbase, []types2.Type{xt}, 1, wantBool); ok {
				return r.makeCall(op.Pos(), op.Y, name, []syntax.Expr{op.X})
			}
		}
	}
	return nil
}

func (r *postTypes2MagicRewriter) makeCall(pos syntax.Pos, recv syntax.Expr, method string, args []syntax.Expr) *syntax.CallExpr {
	sel := &syntax.SelectorExpr{X: recv, Sel: &syntax.Name{Value: method}}
	sel.SetPos(pos)
	sel.Sel.SetPos(pos)
	call := &syntax.CallExpr{Fun: sel, ArgList: args}
	call.SetPos(pos)
	return call
}

// ---- index / setitem lowering ----

func (r *postTypes2MagicRewriter) splitIndexArgs(index syntax.Expr) (args []syntax.Expr, hasComma bool, ok bool) {
	if index == nil {
		return nil, false, false
	}
	// Comma: ListExpr
	if l, _ := index.(*syntax.ListExpr); l != nil {
		if len(l.ElemList) == 0 {
			return nil, true, false
		}
		for _, e := range l.ElemList {
			if !r.isValueExpr(e) {
				// Likely type instantiation list; don't touch.
				return nil, true, false
			}
		}
		return l.ElemList, true, true
	}
	if !r.isValueExpr(index) {
		return nil, false, false
	}
	return []syntax.Expr{index}, false, true
}

func (r *postTypes2MagicRewriter) tryRewriteIndexToGetItem(ix *syntax.IndexExpr) *syntax.CallExpr {
	if ix == nil {
		return nil
	}
	recvT := r.typeOf(ix.X)
	if recvT == nil {
		return nil
	}
	// Do not rewrite native indexable types (or named types with such underlyings).
	// These must keep Go's built-in indexing semantics (incl. assignability and comma-ok).
	switch ct := types2.CoreType(recvT).(type) {
	case *types2.Basic, *types2.Slice, *types2.Array, *types2.Map:
		return nil
	case *types2.Pointer:
		// Keep native *[N]T indexing unchanged, but allow pointer receivers otherwise.
		if _, ok := types2.CoreType(ct.Elem()).(*types2.Array); ok {
			return nil
		}
	}

	args, hasComma, ok := r.splitIndexArgs(ix.Index)
	if !ok {
		return nil
	}

	// Decide between slice-style (wrap each comma elem into []int{...}) vs flat args.
	callArgs := args
	argTypes := make([]types2.Type, 0, len(callArgs))
	for _, a := range callArgs {
		if t := r.typeOf(a); t != nil {
			argTypes = append(argTypes, t)
		}
	}

	// First try direct (as-is) matching.
	if name, _, ok := r.lookupBestMagic(recvT, "_getitem", argTypes, 1, false); ok {
		return r.makeCall(ix.Pos(), ix.X, name, callArgs)
	}

	// If comma syntax, try wrapping SliceExpr{X:nil} into []int{...} (typechecks as []int),
	// and try flattening into scalar args.
	if hasComma {
		wrapped := make([]syntax.Expr, 0, len(args))
		for _, a := range args {
			wrapped = append(wrapped, r.wrapCommaElemToIntSlice(a))
		}
		wrappedTypes := make([]types2.Type, 0, len(wrapped))
		for range wrapped {
			// These are new nodes; best-effort type is []int.
			wrappedTypes = append(wrappedTypes, types2.NewSlice(types2.Typ[types2.Int]))
		}
		if name, _, ok := r.lookupBestMagic(recvT, "_getitem", wrappedTypes, 1, false); ok {
			return r.makeCall(ix.Pos(), ix.X, name, wrapped)
		}

		flat := r.flattenCommaElems(args)
		flatTypes := make([]types2.Type, 0, len(flat))
		for _, a := range flat {
			if t := r.typeOf(a); t != nil {
				flatTypes = append(flatTypes, t)
			}
		}
		if name, _, ok := r.lookupBestMagic(recvT, "_getitem", flatTypes, 1, false); ok {
			return r.makeCall(ix.Pos(), ix.X, name, flat)
		}
	}

	return nil
}

func (r *postTypes2MagicRewriter) tryRewriteSliceToGetItem(se *syntax.SliceExpr) *syntax.CallExpr {
	if se == nil || se.X == nil {
		return nil
	}
	recvT := r.typeOf(se.X)
	if recvT == nil {
		return nil
	}
	// Keep native slicing semantics for builtin slice/array/string/pointer-to-array.
	switch ct := types2.CoreType(recvT).(type) {
	case *types2.Basic, *types2.Slice, *types2.Array:
		return nil
	case *types2.Pointer:
		if _, ok := types2.CoreType(ct.Elem()).(*types2.Array); ok {
			return nil
		}
	}
	args := make([]syntax.Expr, 0, 3)
	argTypes := make([]types2.Type, 0, 3)
	for _, idx := range se.Index {
		if idx == nil {
			continue
		}
		if !r.isValueExpr(idx) {
			return nil
		}
		args = append(args, idx)
		if t := r.typeOf(idx); t != nil {
			argTypes = append(argTypes, t)
		}
	}
	if len(args) == 0 {
		return nil
	}
	if name, _, ok := r.lookupBestMagic(recvT, "_getitem", argTypes, 1, false); ok {
		return r.makeCall(se.Pos(), se.X, name, args)
	}
	return nil
}

func (r *postTypes2MagicRewriter) tryRewriteIndexAssignToSetItem(pos syntax.Pos, lhs, rhs syntax.Expr) *syntax.CallExpr {
	var base syntax.Expr
	var idxArgs []syntax.Expr
	var hasComma bool

	switch l := lhs.(type) {
	case *syntax.IndexExpr:
		base = l.X
		args, comma, ok := r.splitIndexArgs(l.Index)
		if !ok {
			return nil
		}
		idxArgs, hasComma = args, comma
	case *syntax.SliceExpr:
		if l.X == nil {
			return nil
		}
		base = l.X
		for _, idx := range l.Index {
			if idx != nil {
				idxArgs = append(idxArgs, idx)
			}
		}
	default:
		return nil
	}

	recvT := r.typeOf(base)
	if recvT == nil {
		return nil
	}
	// Do not rewrite assignments on native indexable types; they must remain addressable LHS.
	switch ct := types2.CoreType(recvT).(type) {
	case *types2.Basic, *types2.Slice, *types2.Array, *types2.Map:
		return nil
	case *types2.Pointer:
		// Keep native *[N]T indexing assignments unchanged.
		if _, ok := types2.CoreType(ct.Elem()).(*types2.Array); ok {
			return nil
		}
	}

	// value-first convention: _setitem(value, idx...)
	valueArg := rhs

	// Attempt direct match first.
	callArgs := append([]syntax.Expr{valueArg}, idxArgs...)
	argTypes := make([]types2.Type, 0, len(callArgs))
	for _, a := range callArgs {
		if t := r.typeOf(a); t != nil {
			argTypes = append(argTypes, t)
		}
	}
	if name, _, ok := r.lookupBestMagic(recvT, "_setitem", argTypes, 0, false); ok {
		return r.makeCall(pos, base, name, callArgs)
	}

	if hasComma {
		// Try slice-style wrapping for indices only.
		wrappedIdx := make([]syntax.Expr, 0, len(idxArgs))
		for _, a := range idxArgs {
			wrappedIdx = append(wrappedIdx, r.wrapCommaElemToIntSlice(a))
		}
		wrappedArgs := append([]syntax.Expr{valueArg}, wrappedIdx...)
		wrappedTypes := make([]types2.Type, 0, len(wrappedArgs))
		// First arg type from info; index args best-effort as []int.
		if t := r.typeOf(valueArg); t != nil {
			wrappedTypes = append(wrappedTypes, t)
		}
		for range wrappedIdx {
			wrappedTypes = append(wrappedTypes, types2.NewSlice(types2.Typ[types2.Int]))
		}
		if name, _, ok := r.lookupBestMagic(recvT, "_setitem", wrappedTypes, 0, false); ok {
			return r.makeCall(pos, base, name, wrappedArgs)
		}

		flatIdx := r.flattenCommaElems(idxArgs)
		flatArgs := append([]syntax.Expr{valueArg}, flatIdx...)
		flatTypes := make([]types2.Type, 0, len(flatArgs))
		for _, a := range flatArgs {
			if t := r.typeOf(a); t != nil {
				flatTypes = append(flatTypes, t)
			}
		}
		if name, _, ok := r.lookupBestMagic(recvT, "_setitem", flatTypes, 0, false); ok {
			return r.makeCall(pos, base, name, flatArgs)
		}
	}

	return nil
}

func (r *postTypes2MagicRewriter) wrapCommaElemToIntSlice(elem syntax.Expr) syntax.Expr {
	// elem may be SliceExpr{X:nil} meaning "a:b" inside comma index list.
	var parts []syntax.Expr
	if se, _ := elem.(*syntax.SliceExpr); se != nil && se.X == nil {
		for _, idx := range se.Index {
			if idx != nil {
				parts = append(parts, idx)
			}
		}
	} else {
		parts = []syntax.Expr{elem}
	}
	if len(parts) == 0 {
		return elem
	}
	pos := parts[0].Pos()
	// Build type: []int
	st := &syntax.SliceType{Elem: &syntax.Name{Value: "int"}}
	st.SetPos(pos)
	st.Elem.(*syntax.Name).SetPos(pos)
	lit := &syntax.CompositeLit{Type: st, ElemList: parts}
	lit.SetPos(pos)
	return lit
}

func (r *postTypes2MagicRewriter) flattenCommaElems(elems []syntax.Expr) []syntax.Expr {
	var out []syntax.Expr
	for _, elem := range elems {
		if se, _ := elem.(*syntax.SliceExpr); se != nil && se.X == nil {
			for _, idx := range se.Index {
				if idx != nil {
					out = append(out, idx)
				}
			}
			continue
		}
		out = append(out, elem)
	}
	return out
}

// ---- constructor make(T, ...) lowering ----

func (r *postTypes2MagicRewriter) tryRewriteMakeToInit(call *syntax.CallExpr) *syntax.CallExpr {
	if call == nil || call.HasDots || len(call.ArgList) == 0 {
		return nil
	}
	// Only handle builtin make(...) calls.
	if !r.typeAndValue(call.Fun).IsBuiltin() {
		return nil
	}
	// First argument must be a type.
	arg0 := call.ArgList[0]
	tv0 := r.typeAndValue(arg0)
	if !tv0.IsType() {
		return nil
	}
	T := tv0.Type
	if T == nil {
		return nil
	}
	// Never rewrite builtin make for native slice/map/chan. Those must stay as builtin make.
	switch types2.CoreType(T).(type) {
	case *types2.Slice, *types2.Map, *types2.Chan:
		return nil
	}

	// Only rewrite "constructor make" forms: make(T, args...) where T has _init.
	ptrT := types2.NewPointer(T)
	argTypes := make([]types2.Type, 0, len(call.ArgList)-1)
	for _, a := range call.ArgList[1:] {
		if t := r.typeOf(a); t != nil {
			argTypes = append(argTypes, t)
		}
	}
	// _init(...) is expected to return *T so make(T, ...) is an expression.
	name, _, ok := r.lookupBestMagic(ptrT, "_init", argTypes, 1, false)
	if !ok {
		return nil
	}

	pos := call.Pos()
	// Build: (&T{})._init(args...)
	cl := &syntax.CompositeLit{Type: arg0}
	cl.SetPos(pos)
	addr := &syntax.Operation{Op: syntax.And, X: cl}
	addr.SetPos(pos)
	par := &syntax.ParenExpr{X: addr}
	par.SetPos(pos)
	sel := &syntax.SelectorExpr{X: par, Sel: &syntax.Name{Value: name}}
	sel.SetPos(pos)
	sel.Sel.SetPos(pos)
	newCall := &syntax.CallExpr{Fun: sel, ArgList: call.ArgList[1:], HasDots: call.HasDots}
	newCall.SetPos(pos)
	return newCall
}
