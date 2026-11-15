// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"slices"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func writeConsts(pkg *types2.Package, info *types2.Info, files []*syntax.File, f *Facts) {
	if pkg == nil || pkg.Scope() == nil {
		return
	}
	emitConstDecls(pkg, f)
	emitConstMutations(pkg, info, files, f)
}

func emitConstDecls(pkg *types2.Package, f *Facts) {
	emit := func(object, path string) {
		f.Consts = appendUnique(f.Consts, PathFact{Object: object, Path: path})
	}
	eachExportedSig(pkg, func(object string, sig *types2.Signature) {
		walkSigQuals(sig, object, nil, emit)
	})
	eachExportedNamed(pkg, func(name string, named *types2.Named) {
		walkNamedQuals(name, named, nil, emit)
	})
}

type constFuncDecl struct {
	obj  *types2.Func
	decl *syntax.FuncDecl
	sig  *types2.Signature
	name string
}

func emitConstMutations(pkg *types2.Package, info *types2.Info, files []*syntax.File, f *Facts) {
	decls, recvMut, paramMask := analyzeConstMutations(pkg, info, files)
	for _, d := range decls {
		if !recvMut[d.obj] && len(paramMask[d.obj]) == 0 {
			continue
		}
		fn := findOrCreateFnFact(f, d.name)
		fn.ReceiverMutated = fn.ReceiverMutated || recvMut[d.obj]
		fn.ParamMutated = orHexMask(fn.ParamMutated, paramMask[d.obj])
	}
	emitReadonlyFacts(pkg, f, decls, recvMut, paramMask)
}

func analyzeConstMutations(pkg *types2.Package, info *types2.Info, files []*syntax.File) ([]constFuncDecl, map[*types2.Func]bool, map[*types2.Func][]byte) {
	return analyzeMutations(pkg, info, files, false)
}

func analyzeGoMutations(pkg *types2.Package, info *types2.Info, files []*syntax.File) ([]constFuncDecl, map[*types2.Func]bool, map[*types2.Func][]byte) {
	return analyzeMutations(pkg, info, files, true)
}

func analyzeMutations(pkg *types2.Package, info *types2.Info, files []*syntax.File, strictRetain bool) ([]constFuncDecl, map[*types2.Func]bool, map[*types2.Func][]byte) {
	decls := collectConstFuncDecls(pkg, info, files)
	recvMut := make(map[*types2.Func]bool, len(decls))
	paramMask := make(map[*types2.Func][]byte, len(decls))
	for _, d := range decls {
		paramMask[d.obj] = []byte{}
	}
	for range len(decls) + 1 {
		changed := false
		for _, d := range decls {
			scan := &constBodyScan{pkg: pkg, info: info, recvMut: recvMut, paramMask: paramMask, strictRetain: strictRetain}
			rm, mask := scan.funcMutations(d)
			rm = rm && !d.obj.LogosReadonly()
			if rm && !recvMut[d.obj] {
				recvMut[d.obj] = true
				changed = true
			}
			if !bytes.Equal(mask, paramMask[d.obj]) {
				paramMask[d.obj] = mask
				changed = true
			}
		}
		if !changed {
			break
		}
	}
	return decls, recvMut, paramMask
}

func EmitGoMutations(pkg *types2.Package, info *types2.Info, files []*syntax.File, f *Facts) *Facts {
	if pkg == nil || pkg.Scope() == nil {
		return f
	}
	decls, recvMut, paramMask := analyzeGoMutations(pkg, info, files)
	for _, d := range decls {
		if d.obj == nil || !d.obj.Exported() {
			continue
		}
		if d.sig == nil || d.sig.Params() == nil || d.sig.Params().Len() == 0 {
			continue
		}
		mask := paramMask[d.obj]
		if f == nil {
			f = &Facts{}
		}
		fn := findOrCreateFnFact(f, d.name)
		fn.ReceiverMutated = fn.ReceiverMutated || recvMut[d.obj]
		fn.ParamMutated = orHexMask(fn.ParamMutated, mask)
		if fn.ParamMutated == "" {
			fn.ParamMutated = "0"
		}
	}
	return f
}

func emitReadonlyFacts(pkg *types2.Package, f *Facts, decls []constFuncDecl, recvMut map[*types2.Func]bool, paramMask map[*types2.Func][]byte) {
	for _, d := range decls {
		if d.obj == nil || !d.obj.LogosReadonly() {
			continue
		}
		if recvMut[d.obj] || len(paramMask[d.obj]) != 0 {
			continue
		}
		fn := findOrCreateFnFact(f, d.name)
		fn.Readonly = true
		if fn.ParamMutated == "" {
			fn.ParamMutated = "0"
		}
	}
	scope := pkg.Scope()
	if scope == nil {
		return
	}
	for _, name := range scope.Names() {
		obj, ok := scope.Lookup(name).(*types2.TypeName)
		if !ok || obj == nil || !obj.Exported() || obj.IsAlias() {
			continue
		}
		named, ok := obj.Type().(*types2.Named)
		if !ok {
			continue
		}
		u, ok := named.Underlying().(*types2.Interface)
		if !ok {
			continue
		}
		for i := range u.NumExplicitMethods() {
			m := u.ExplicitMethod(i)
			if m == nil || !m.Exported() {
				continue
			}
			if !m.LogosReadonly() {
				continue
			}
			findOrCreateFnFact(f, name+"."+m.Name()).Readonly = true
		}
	}
	emitReadonlyNestedFacts(pkg, f)
}

func emitReadonlyNestedFacts(pkg *types2.Package, f *Facts) {
	emit := func(object, path string) {
		f.Readonlys = appendUnique(f.Readonlys, ReadonlyFact{Object: object, Path: path})
	}
	scope := pkg.Scope()
	if scope == nil {
		return
	}
	for _, name := range scope.Names() {
		obj := scope.Lookup(name)
		if obj == nil || !obj.Exported() {
			continue
		}
		switch obj := obj.(type) {
		case *types2.Func:
			sig, ok := obj.Type().(*types2.Signature)
			if !ok || sig.Recv() != nil {
				continue
			}
			walkReadonlySig(sig, nil, name, emit, make(map[types2.Type]bool))
		case *types2.TypeName:
			walkReadonlyType(obj.Type(), nil, name, emit, make(map[types2.Type]bool))
		}
	}
	for _, name := range scope.Names() {
		obj, ok := scope.Lookup(name).(*types2.TypeName)
		if !ok || obj == nil || !obj.Exported() || obj.IsAlias() {
			continue
		}
		named, ok := obj.Type().(*types2.Named)
		if !ok {
			continue
		}
		for i := range named.NumMethods() {
			m := named.Method(i)
			if m == nil || !m.Exported() {
				continue
			}
			sig, ok := m.Type().(*types2.Signature)
			if !ok {
				continue
			}
			walkReadonlySig(sig, nil, MethodObject(name, sig.Recv(), m.Name()), emit, make(map[types2.Type]bool))
		}
	}
}

func walkReadonlySig(sig *types2.Signature, base []string, object string, emit func(string, string), seen map[types2.Type]bool) {
	if sig == nil {
		return
	}
	if tp := sig.TypeParams(); tp != nil {
		for i := range tp.Len() {
			p := tp.At(i)
			if p == nil {
				continue
			}
			walkReadonlyType(p.Constraint(), append(slices.Clone(base), fmt.Sprintf("G%d", i)), object, emit, seen)
		}
	}
	if rp := sig.RecvTypeParams(); rp != nil {
		for i := range rp.Len() {
			p := rp.At(i)
			if p == nil {
				continue
			}
			walkReadonlyType(p.Constraint(), append(slices.Clone(base), fmt.Sprintf("H%d", i)), object, emit, seen)
		}
	}
	if params := sig.Params(); params != nil {
		for i := range params.Len() {
			v := params.At(i)
			if v == nil {
				continue
			}
			walkReadonlyType(v.Type(), append(slices.Clone(base), fmt.Sprintf("P%d", i)), object, emit, seen)
		}
	}
	if results := sig.Results(); results != nil {
		for i := range results.Len() {
			v := results.At(i)
			if v == nil {
				continue
			}
			walkReadonlyType(v.Type(), append(slices.Clone(base), fmt.Sprintf("R%d", i)), object, emit, seen)
		}
	}
}

func walkReadonlyType(t types2.Type, path []string, object string, emit func(string, string), seen map[types2.Type]bool) {
	if t == nil {
		return
	}
	if a, ok := t.(*types2.Alias); ok {
		if tp := a.TypeParams(); tp != nil {
			for i := range tp.Len() {
				p := tp.At(i)
				if p == nil {
					continue
				}
				walkReadonlyType(p.Constraint(), append(slices.Clone(path), fmt.Sprintf("G%d", i)), object, emit, seen)
			}
		}
		walkReadonlyType(a.Rhs(), path, object, emit, seen)
		return
	}
	if opt, ok := unaliasAs[*types2.Option](t); ok {
		walkReadonlyType(opt.Elem, append(slices.Clone(path), "E"), object, emit, seen)
		return
	}
	if c, ok := unaliasAs[*types2.ConstView](t); ok {
		walkReadonlyType(c.Elem, append(slices.Clone(path), "E"), object, emit, seen)
		return
	}
	if p, ok := t.(*types2.TypeParam); ok {
		if seen[p] {
			return
		}
		seen[p] = true
		defer delete(seen, p)
		walkReadonlyType(p.Constraint(), path, object, emit, seen)
		return
	}
	u := types2.Unalias(t)
	switch u := u.(type) {
	case *types2.Pointer:
		walkReadonlyType(u.Elem(), append(slices.Clone(path), "D"), object, emit, seen)
	case *types2.Chan:
		walkReadonlyType(u.Elem(), append(slices.Clone(path), "D"), object, emit, seen)
	case *types2.Slice:
		walkReadonlyType(u.Elem(), append(slices.Clone(path), "E"), object, emit, seen)
	case *types2.Array:
		walkReadonlyType(u.Elem(), append(slices.Clone(path), "E"), object, emit, seen)
	case *types2.Map:
		walkReadonlyType(u.Key(), append(slices.Clone(path), "K"), object, emit, seen)
		walkReadonlyType(u.Elem(), append(slices.Clone(path), "E"), object, emit, seen)
	case *types2.Signature:
		walkReadonlySig(u, path, object, emit, seen)
	case *types2.Struct:
		for i := range u.NumFields() {
			f := u.Field(i)
			if f == nil || !f.Exported() {
				continue
			}
			walkReadonlyType(f.Type(), append(slices.Clone(path), "F"+f.Name()), object, emit, seen)
		}
	case *types2.Interface:
		for i := range u.NumExplicitMethods() {
			m := u.ExplicitMethod(i)
			if m == nil {
				continue
			}
			if m.LogosReadonly() {
				emit(object, joinPath(append(slices.Clone(path), "M"+m.Name())))
			}
			sig, ok := m.Type().(*types2.Signature)
			if !ok {
				continue
			}
			walkReadonlySig(sig, append(slices.Clone(path), "M"+m.Name()), object, emit, seen)
		}
		for i := range u.NumEmbeddeds() {
			walkReadonlyType(u.EmbeddedType(i), path, object, emit, seen)
		}
	case *types2.Named:
		if seen[u] {
			return
		}
		seen[u] = true
		defer delete(seen, u)
		if tp := u.TypeParams(); tp != nil {
			for i := range tp.Len() {
				p := tp.At(i)
				if p == nil {
					continue
				}
				walkReadonlyType(p.Constraint(), append(slices.Clone(path), fmt.Sprintf("G%d", i)), object, emit, seen)
			}
		}
		walkReadonlyType(u.Underlying(), path, object, emit, seen)
	case *types2.Tuple:
		for i := range u.Len() {
			v := u.At(i)
			if v == nil {
				continue
			}
			walkReadonlyType(v.Type(), append(slices.Clone(path), fmt.Sprintf("T%d", i)), object, emit, seen)
		}
	case *types2.Union:
		for i := range u.Len() {
			tm := u.Term(i)
			if tm == nil {
				continue
			}
			walkReadonlyType(tm.Type(), path, object, emit, seen)
		}
	}
}

func orHexMask(have string, add []byte) string {
	if len(add) == 0 {
		return have
	}
	hb, _ := parseHexMask(have)
	return maskHex(orMask(hb, add))
}

func parseHexMask(s string) ([]byte, bool) {
	if s == "" {
		return nil, false
	}
	if len(s)%2 != 0 {
		s = "0" + s
	}
	b, err := hex.DecodeString(s)
	if err != nil {
		return nil, false
	}
	b = bytes.TrimLeft(b, "\x00")
	if len(b) == 0 {
		return []byte{}, true
	}
	return b, true
}

func maskBit(b []byte, i int) bool {
	if i < 0 || i/8 >= len(b) {
		return false
	}
	return b[len(b)-1-i/8]>>(uint(i)%8)&1 != 0
}

func setMaskBit(b []byte, i int) []byte {
	if i < 0 {
		return b
	}
	if i/8 >= len(b) {
		b = append(make([]byte, i/8+1-len(b)), b...)
	}
	b[len(b)-1-i/8] |= 1 << (uint(i) % 8)
	return b
}

func orMask(a, b []byte) []byte {
	n := len(a)
	if len(b) > n {
		n = len(b)
	}
	out := make([]byte, n)
	copy(out[n-len(a):], a)
	for i := range b {
		out[n-len(b)+i] |= b[i]
	}
	return out
}

func maskHex(b []byte) string {
	if s := hex.EncodeToString(bytes.TrimLeft(b, "\x00")); s != "" {
		if len(s) > 1 && s[0] == '0' {
			s = s[1:]
		}
		return s
	}
	return "0"
}

func collectConstFuncDecls(pkg *types2.Package, info *types2.Info, files []*syntax.File) []constFuncDecl {
	var decls []constFuncDecl
	for _, file := range files {
		for _, decl := range file.DeclList {
			fd, ok := decl.(*syntax.FuncDecl)
			if !ok || fd.Body == nil {
				continue
			}
			obj, ok := lookupConstFunc(info, fd, pkg)
			if !ok {
				continue
			}
			sig, ok := obj.Type().(*types2.Signature)
			if !ok {
				continue
			}
			if tp := sig.TypeParams(); tp != nil && tp.Len() > 0 {
				continue
			}
			decls = append(decls, constFuncDecl{obj: obj, decl: fd, sig: sig, name: constFuncName(pkg, obj, sig)})
		}
	}
	return decls
}

func lookupConstFunc(info *types2.Info, fd *syntax.FuncDecl, pkg *types2.Package) (*types2.Func, bool) {
	if info == nil {
		return nil, false
	}
	obj, ok := info.Defs[fd.Name].(*types2.Func)
	if !ok || obj == nil || obj.Pkg() != pkg {
		return nil, false
	}
	return obj, true
}

func constRecvBase(t types2.Type) types2.Type {
	t = types2.Unalias(t)
	if c, ok := t.(*types2.ConstView); ok {
		t = types2.Unalias(c.Elem)
	}
	if ptr, ok := t.(*types2.Pointer); ok {
		t = types2.Unalias(ptr.Elem())
		if c, ok := t.(*types2.ConstView); ok {
			t = types2.Unalias(c.Elem)
		}
	}
	return t
}

func constFuncName(pkg *types2.Package, obj *types2.Func, sig *types2.Signature) string {
	if sig.Recv() == nil {
		return obj.Name()
	}
	if named, ok := constRecvBase(sig.Recv().Type()).(*types2.Named); ok && named.Obj().Pkg() == pkg {
		return MethodObject(named.Obj().Name(), sig.Recv(), obj.Name())
	}
	return obj.Name()
}

type constBodyScan struct {
	pkg          *types2.Package
	info         *types2.Info
	recvMut      map[*types2.Func]bool
	paramMask    map[*types2.Func][]byte
	tainted      map[types2.Object]*types2.Var
	mut          map[*types2.Var]bool
	strictRetain bool
	results      []*types2.Var
	slots        []types2.Type
}

func (s *constBodyScan) funcMutations(d constFuncDecl) (bool, []byte) {
	s.tainted = make(map[types2.Object]*types2.Var)
	s.mut = make(map[*types2.Var]bool)
	s.results = nil
	var params []*types2.Var
	recv := d.sig.Recv()
	if recv != nil {
		params = append(params, recv)
	}
	if ps := d.sig.Params(); ps != nil {
		for i := range ps.Len() {
			params = append(params, ps.At(i))
		}
	}
	for _, p := range params {
		if p != nil {
			s.tainted[p] = p
		}
	}
	s.slots = nil
	if rs := d.sig.Results(); rs != nil {
		for i := range rs.Len() {
			if v := rs.At(i); v != nil {
				s.results = append(s.results, v)
				s.slots = append(s.slots, v.Type())
			}
		}
	}
	s.block(d.decl.Body.List)
	mask := []byte{}
	bit := 0
	for _, p := range params {
		if p == recv {
			continue
		}
		if s.mut[p] {
			mask = setMaskBit(mask, bit)
		}
		bit++
	}
	return recv != nil && s.mut[recv], mask
}

func (s *constBodyScan) resolve(name *syntax.Name) types2.Object {
	if name == nil || s.info == nil {
		return nil
	}
	if obj, ok := s.info.Defs[name]; ok && obj != nil {
		return obj
	}
	return s.info.Uses[name]
}

func (s *constBodyScan) isPackageVar(obj types2.Object) bool {
	v, ok := obj.(*types2.Var)
	if !ok || v == nil || s.pkg == nil {
		return false
	}
	return s.pkg.Scope().Lookup(v.Name()) == types2.Object(v)
}

func (s *constBodyScan) originOf(obj types2.Object) *types2.Var {
	if obj == nil {
		return nil
	}
	return s.tainted[obj]
}

func (s *constBodyScan) mark(obj types2.Object) {
	if o := s.originOf(obj); o != nil {
		if t := o.Type(); t != nil && types2.LogosReferenceFree(t) {
			return
		}
		s.mut[o] = true
	}
}

func (s *constBodyScan) markWrite(obj types2.Object) {
	if o := s.originOf(obj); o != nil {
		s.mut[o] = true
	}
}

func (s *constBodyScan) writeCrossesRef(e syntax.Expr) bool {
	e = syntax.Unparen(e)
	switch n := e.(type) {
	case *syntax.IndexExpr:
		t := s.info.TypeOf(n.X)
		if t == nil {
			return true
		}
		switch types2.Unalias(t).Underlying().(type) {
		case *types2.Slice, *types2.Map:
			return true
		case *types2.Array:
			return s.writeCrossesRef(n.X)
		default:
			return true
		}
	case *syntax.SliceExpr:
		return true
	case *syntax.Operation:
		return n.Op == syntax.Mul && n.Y == nil
	case *syntax.SelectorExpr:
		t := s.info.TypeOf(n.X)
		if t == nil {
			return true
		}
		if _, ok := types2.Unalias(t).(*types2.Pointer); ok {
			return true
		}
		return s.writeCrossesRef(n.X)
	case *syntax.ListExpr:
		for _, el := range n.ElemList {
			if ex, ok := el.(syntax.Expr); ok && s.writeCrossesRef(ex) {
				return true
			}
		}
		return false
	default:
		return false
	}
}

func (s *constBodyScan) markMentionWrite(e syntax.Node) {
	syntax.Inspect(e, func(n syntax.Node) bool {
		if idx, ok := n.(*syntax.IndexExpr); ok {
			s.markMentionWrite(idx.X)
			s.scanReads(idx.Index)
			return false
		}
		if sl, ok := n.(*syntax.SliceExpr); ok {
			s.markMentionWrite(sl.X)
			for _, bound := range sl.Index {
				if bound != nil {
					s.scanReads(bound)
				}
			}
			return false
		}
		if name, ok := n.(*syntax.Name); ok {
			s.markWrite(s.resolve(name))
			return true
		}
		if call, ok := n.(*syntax.CallExpr); ok {
			s.callEscaping(call)
			return false
		}
		if lit, ok := n.(*syntax.FuncLit); ok {
			s.markClosure(lit.Body)
			return false
		}
		return true
	})
}

func (s *constBodyScan) taintedMention(e syntax.Expr) *types2.Var {
	var found *types2.Var
	syntax.Inspect(e, func(n syntax.Node) bool {
		if found != nil {
			return false
		}
		name, ok := n.(*syntax.Name)
		if !ok {
			return true
		}
		found = s.originOf(s.resolve(name))
		return found == nil
	})
	return found
}

func (s *constBodyScan) block(list []syntax.Stmt) {
	for _, st := range list {
		s.stmt(st)
	}
}

func (s *constBodyScan) stmt(st syntax.Stmt) {
	switch st := st.(type) {
	case *syntax.AssignStmt:
		s.assign(st)
	case *syntax.ReturnStmt:
		if st.Results != nil {
			s.returned(st.Results)
			break
		}
		for _, r := range s.results {
			if s.originOf(r) != nil {
				s.mark(r)
			}
		}
	case *syntax.SendStmt:
		s.markMention(st.Value)
	case *syntax.CallStmt:
		s.suspended(st)
	case *syntax.DeclStmt:
		syntax.Inspect(st, func(n syntax.Node) bool {
			if vd, ok := n.(*syntax.VarDecl); ok {
				s.varDecl(vd)
				return false
			}
			return true
		})
	case *syntax.IfStmt:
		if st.Init != nil {
			s.stmt(st.Init)
		}
		if st.Cond != nil {
			s.scanReads(st.Cond)
		}
		s.blockStmt(st.Then)
		if st.Else != nil {
			s.stmt(st.Else)
		}
	case *syntax.ForStmt:
		if st.Init != nil {
			if rc, ok := st.Init.(*syntax.RangeClause); ok {
				s.rangeTaint(rc)
				s.scanReads(st.Init)
			} else {
				s.stmt(st.Init)
			}
		}
		if st.Cond != nil {
			s.scanReads(st.Cond)
		}
		if st.Post != nil {
			s.stmt(st.Post)
		}
		s.blockStmt(st.Body)
	case *syntax.SwitchStmt:
		if st.Init != nil {
			s.stmt(st.Init)
		}
		if st.Tag != nil {
			s.scanReads(st.Tag)
		}
		for _, c := range st.Body {
			s.block(c.Body)
		}
	case *syntax.SelectStmt:
		for _, c := range st.Body {
			if c.Comm != nil {
				s.stmt(c.Comm)
			}
			s.block(c.Body)
		}
	case *syntax.LabeledStmt:
		s.stmt(st.Stmt)
	case *syntax.BlockStmt:
		s.block(st.List)
	case *syntax.EmptyStmt, *syntax.BranchStmt, *syntax.ExprStmt:
		s.scanReads(st)
	default:
		s.markAllParams()
	}
}

func (s *constBodyScan) blockStmt(n syntax.Node) {
	switch n := n.(type) {
	case *syntax.BlockStmt:
		s.block(n.List)
	case syntax.Stmt:
		s.stmt(n)
	}
}

func (s *constBodyScan) returned(results syntax.Expr) {
	exprs := syntax.UnpackListExpr(results)
	for i, e := range exprs {
		if types2.LogosAliasFreeReturn(s.info.TypeOf(e), s.resultType(i, len(exprs))) {
			s.scanReads(e)
			continue
		}
		s.markMention(e)
	}
}

func (s *constBodyScan) resultType(i, n int) types2.Type {
	if n != len(s.slots) {
		return nil
	}
	return s.slots[i]
}

func (s *constBodyScan) markAllParams() {
	for obj := range s.tainted {
		s.mark(obj)
	}
}

func (s *constBodyScan) rangeTaint(rc *syntax.RangeClause) {
	if rc == nil || rc.Lhs == nil || rc.X == nil {
		return
	}
	origin := s.taintedMention(rc.X)
	if origin == nil {
		return
	}
	taint := func(e syntax.Expr) {
		name, ok := syntax.Unparen(e).(*syntax.Name)
		if !ok || name.Value == "_" {
			return
		}
		if obj := s.resolve(name); obj != nil {
			s.tainted[obj] = origin
		}
	}
	if ll, ok := syntax.Unparen(rc.Lhs).(*syntax.ListExpr); ok {
		for _, e := range ll.ElemList {
			taint(e)
		}
		return
	}
	taint(rc.Lhs)
}

func (s *constBodyScan) assign(st *syntax.AssignStmt) {
	if st.Rhs == nil {
		if ex, ok := syntax.Unparen(st.Lhs).(syntax.Expr); ok && s.writeCrossesRef(ex) {
			s.markMentionWrite(st.Lhs)
		} else {
			s.markMention(st.Lhs)
		}
		return
	}
	lhs, _ := st.Lhs.(*syntax.Name)
	if st.Op == 0 || st.Op == syntax.Def {
		if lhs != nil {
			obj := s.resolve(lhs)
			if s.isPackageVar(obj) {
				s.markMention(st.Rhs)
				return
			}
			s.scanReads(st.Rhs)
			if o := s.taintedMention(st.Rhs); o != nil {
				if obj != nil {
					if t := s.info.TypeOf(st.Rhs); t == nil || !types2.LogosReferenceFree(t) {
						s.tainted[obj] = o
					}
				} else if lhs.Value != "_" {
					s.markMention(st.Rhs)
				}
			}
			return
		}
		if ll, ok := syntax.Unparen(st.Lhs).(*syntax.ListExpr); ok {
			s.assignList(ll, st.Rhs)
			return
		}
		if ex, ok := syntax.Unparen(st.Lhs).(syntax.Expr); ok && s.writeCrossesRef(ex) {
			s.markMentionWrite(st.Lhs)
		} else {
			s.markMention(st.Lhs)
		}
		if s.mentionsPackage(st.Lhs) {
			s.markMention(st.Rhs)
		} else if s.strictRetain && s.retainRHS(st.Lhs, st.Rhs) {
			s.markMention(st.Rhs)
		} else {
			s.scanReads(st.Rhs)
		}
		return
	}
	if ex, ok := syntax.Unparen(st.Lhs).(syntax.Expr); ok && s.writeCrossesRef(ex) {
		s.markMentionWrite(st.Lhs)
	} else {
		s.markMention(st.Lhs)
	}
	if st.Rhs != nil {
		s.scanReads(st.Rhs)
	}
}

func (s *constBodyScan) assignList(lhs *syntax.ListExpr, rhs syntax.Expr) {
	var elems []syntax.Expr
	if rl, ok := syntax.Unparen(rhs).(*syntax.ListExpr); ok && len(rl.ElemList) == len(lhs.ElemList) {
		elems = rl.ElemList
	}
	for i, le := range lhs.ElemList {
		var re syntax.Expr
		if elems != nil {
			re = elems[i]
		} else {
			re = rhs
		}
		s.assignElem(le, re)
	}
}

func (s *constBodyScan) assignElem(le, re syntax.Expr) {
	if name, ok := syntax.Unparen(le).(*syntax.Name); ok {
		if name.Value == "_" {
			return
		}
		obj := s.resolve(name)
		if s.isPackageVar(obj) {
			if re != nil {
				s.markMention(re)
			}
			return
		}
		if re != nil {
			s.scanReads(re)
			if o := s.taintedMention(re); o != nil && obj != nil {
				if t := s.info.TypeOf(re); t == nil || !types2.LogosReferenceFree(t) {
					s.tainted[obj] = o
				}
			}
		}
		return
	}
	if ex, ok := syntax.Unparen(le).(syntax.Expr); ok && s.writeCrossesRef(ex) {
		s.markMentionWrite(le)
	} else {
		s.markMention(le)
	}
	if re == nil {
		return
	}
	if s.mentionsPackage(le) {
		s.markMention(re)
	} else if s.strictRetain && s.retainRHS(le, re) {
		s.markMention(re)
	} else {
		s.scanReads(re)
	}
}

func (s *constBodyScan) mentionsPackage(e syntax.Expr) bool {
	found := false
	syntax.Inspect(e, func(n syntax.Node) bool {
		if found {
			return false
		}
		if name, ok := n.(*syntax.Name); ok {
			found = s.isPackageVar(s.resolve(name))
		}
		return !found
	})
	return found
}

func (s *constBodyScan) retainRHS(lhs syntax.Expr, rhs syntax.Expr) bool {
	origin := s.taintedMention(rhs)
	if origin == nil {
		return false
	}
	if t := s.info.TypeOf(rhs); t != nil && types2.LogosReferenceFree(t) {
		return false
	}
	obj, ok := s.retentionRoot(lhs)
	if !ok {
		return false
	}
	if obj == nil {
		return true
	}
	if s.originOf(obj) != nil {
		return true
	}
	if v, ok := obj.(*types2.Var); ok && !s.isPackageVar(v) {
		s.tainted[v] = origin
		s.markMention(rhs)
		return true
	}
	return false
}

func (s *constBodyScan) retentionRoot(lhs syntax.Expr) (types2.Object, bool) {
	e := syntax.Unparen(lhs)
	deref := false
	for {
		switch n := e.(type) {
		case *syntax.IndexExpr:
			e = syntax.Unparen(n.X)
		case *syntax.SliceExpr:
			e = syntax.Unparen(n.X)
		case *syntax.SelectorExpr:
			e = syntax.Unparen(n.X)
		case *syntax.Operation:
			if n.Op != syntax.Mul || n.Y != nil {
				return nil, false
			}
			e = syntax.Unparen(n.X)
			deref = true
		case *syntax.CallExpr:
			return nil, true
		default:
			name, ok := e.(*syntax.Name)
			if !ok || name.Value == "_" {
				return nil, false
			}
			obj := s.resolve(name)
			if obj == nil {
				return nil, true
			}
			if _, ok := obj.(*types2.PkgName); ok {
				return nil, true
			}
			if s.originOf(obj) != nil {
				return obj, true
			}
			if _, ok := obj.(*types2.Var); ok {
				if deref {
					return nil, true
				}
				return obj, true
			}
			return nil, true
		}
	}
}

func (s *constBodyScan) varDecl(vd *syntax.VarDecl) {
	if vd.Values == nil {
		return
	}
	s.scanReads(vd.Values)
	origin := s.taintedMention(vd.Values)
	for _, name := range vd.NameList {
		obj := s.resolve(name)
		if obj == nil {
			continue
		}
		if s.isPackageVar(obj) {
			s.markMention(vd.Values)
			continue
		}
		if origin != nil {
			s.tainted[obj] = origin
		}
	}
}

func (s *constBodyScan) doCall(call *syntax.CallExpr, escaping bool) {
	if lit, ok := syntax.Unparen(call.Fun).(*syntax.FuncLit); ok {
		if lit.Body != nil && !escaping {
			slots := s.slots
			s.slots = nil
			s.block(lit.Body.List)
			s.slots = slots
		} else if lit.Body != nil {
			s.markClosure(lit.Body)
		}
		s.markArgs(call.ArgList)
		return
	}
	if name, ok := call.Fun.(*syntax.Name); ok {
		if obj := s.resolve(name); obj != nil {
			if b, ok := obj.(*types2.Builtin); ok {
				s.builtinCall(b.Name(), call)
				return
			}
			if fn, ok := obj.(*types2.Func); ok {
				s.doFuncCall(fn, nil, call.ArgList, escaping)
				return
			}
			if tn, ok := obj.(*types2.TypeName); ok && tn != nil {
				if t := tn.Type(); t != nil && types2.LogosReferenceFree(t) {
					for _, arg := range call.ArgList {
						s.scanReads(arg)
					}
					return
				}
			}
		}
		s.markArgs(call.ArgList)
		return
	}
	if sel, ok := call.Fun.(*syntax.SelectorExpr); ok {
		if s.info != nil {
			if choice, ok := s.info.Selections[sel]; ok && choice != nil {
				if fn, ok := choice.Obj().(*types2.Func); ok {
					recv, rargs := syntax.Expr(sel.X), call.ArgList
					if choice.Kind() == types2.MethodExpr && len(rargs) > 0 {
						recv, rargs = rargs[0], rargs[1:]
					}
					s.doFuncCall(fn, recv, rargs, escaping)
					return
				}
				if choice.Kind() == types2.FieldVal {
					s.markMentionWrite(sel.X)
				}
			}
		}
		s.markArgs(call.ArgList)
		return
	}
	s.markArgs(call.ArgList)
}

func (s *constBodyScan) suspended(st *syntax.CallStmt) {
	if st.Tok == syntax.Go {
		s.markMention(st.Call)
		return
	}
	s.scanReads(st.Call)
}

func (s *constBodyScan) markMention(e syntax.Node) {
	syntax.Inspect(e, func(n syntax.Node) bool {
		if name, ok := n.(*syntax.Name); ok {
			s.mark(s.resolve(name))
			return true
		}
		if call, ok := n.(*syntax.CallExpr); ok {
			s.callEscaping(call)
			return false
		}
		if lit, ok := n.(*syntax.FuncLit); ok {
			s.markClosure(lit.Body)
			return false
		}
		if sel, ok := n.(*syntax.SelectorExpr); ok {
			if s.markMethodValue(sel) {
				return false
			}
		}
		return true
	})
}

func (s *constBodyScan) markMethodValue(sel *syntax.SelectorExpr) bool {
	if s.info == nil {
		return false
	}
	choice, ok := s.info.Selections[sel]
	if !ok || choice == nil {
		return false
	}
	if choice.Kind() != types2.MethodVal {
		if _, ok := choice.Obj().Type().(*types2.Signature); !ok {
			return false
		}
	}
	s.markMentionWrite(sel.X)
	return true
}

func (s *constBodyScan) markClosure(body *syntax.BlockStmt) {
	if body == nil {
		return
	}
	syntax.Inspect(body, func(n syntax.Node) bool {
		if lit, ok := n.(*syntax.FuncLit); ok && lit.Body != nil {
			s.markClosure(lit.Body)
			return false
		}
		if name, ok := n.(*syntax.Name); ok {
			s.mark(s.resolve(name))
		}
		return true
	})
}

func (s *constBodyScan) scanReads(e syntax.Node) {
	syntax.Inspect(e, func(n syntax.Node) bool {
		if call, ok := n.(*syntax.CallExpr); ok {
			s.callInFrame(call)
			return false
		}
		if lit, ok := n.(*syntax.FuncLit); ok {
			s.markClosure(lit.Body)
			return false
		}
		if sel, ok := n.(*syntax.SelectorExpr); ok {
			if s.markMethodValue(sel) {
				return false
			}
		}
		return true
	})
}

func (s *constBodyScan) callInFrame(call *syntax.CallExpr) {
	s.doCall(call, false)
}

func (s *constBodyScan) callEscaping(call *syntax.CallExpr) {
	s.doCall(call, true)
}

func (s *constBodyScan) doFuncCall(fn *types2.Func, recv syntax.Expr, args []syntax.Expr, escaping bool) {
	sig, ok := fn.Type().(*types2.Signature)
	if !ok {
		s.markArgs(args)
		return
	}
	mutRecv, mask := s.calleeFacts(fn)
	if recv != nil {
		if escaping {
			s.markMention(recv)
		} else if mutRecv {
			s.markMentionWrite(recv)
		} else {
			s.scanReads(recv)
		}
	}
	if ps := sig.Params(); ps != nil {
		for i, arg := range args {
			if i >= ps.Len() && !sig.Variadic() {
				s.scanReads(arg)
				break
			}
			j := i
			if sig.Variadic() && ps.Len() > 0 && j >= ps.Len()-1 {
				j = ps.Len() - 1
			}
			if escaping {
				s.markMention(arg)
			} else if mask == nil {
				s.markMention(arg)
			} else if maskBit(mask, j) {
				s.markMentionWrite(arg)
			} else {
				s.scanReads(arg)
			}
		}
	} else {
		s.markArgs(args)
	}
}

func (s *constBodyScan) calleeFacts(fn *types2.Func) (bool, []byte) {
	if fn.Pkg() == nil {
		return true, nil
	}
	if fn.Pkg() == s.pkg {
		if mask, ok := s.paramMask[fn]; ok {
			if mask == nil {
				mask = []byte{}
			}
			return s.recvMut[fn], mask
		}
		return true, nil
	}
	if f, ok := Lookup(fn.Pkg().Path()); ok && f != nil {
		want := fn.Name()
		if sig, ok := fn.Type().(*types2.Signature); ok && sig != nil && sig.Recv() != nil {
			want = constMethodName(fn)
		}
		for i := range f.Fns {
			entry := &f.Fns[i]
			if entry.Object != want {
				continue
			}
			if entry.MayYieldOnly() {
				continue
			}
			if entry.ParamMutated == "" {
				return entry.ReceiverMutated, nil
			}
			mask, ok := parseHexMask(entry.ParamMutated)
			if !ok {
				return entry.ReceiverMutated, nil
			}
			return entry.ReceiverMutated, mask
		}
	}
	return true, nil
}

func constMethodName(fn *types2.Func) string {
	sig, ok := fn.Type().(*types2.Signature)
	if !ok || sig.Recv() == nil {
		return fn.Name()
	}
	if named, ok := constRecvBase(sig.Recv().Type()).(*types2.Named); ok {
		return MethodObject(named.Obj().Name(), sig.Recv(), fn.Name())
	}
	return fn.Name()
}

func (s *constBodyScan) builtinCall(name string, call *syntax.CallExpr) {
	switch name {
	case "append", "copy":
		if len(call.ArgList) > 0 {
			s.markMentionWrite(call.ArgList[0])
		}
		for _, arg := range call.ArgList[1:] {
			s.markMention(arg)
		}
	case "delete", "clear":
		if len(call.ArgList) > 0 {
			s.markMentionWrite(call.ArgList[0])
			s.scanReads(call.ArgList[0])
			for _, arg := range call.ArgList[1:] {
				s.scanReads(arg)
			}
		}
	case "panic":
		for _, arg := range call.ArgList {
			s.markMention(arg)
		}
	default:
		for _, arg := range call.ArgList {
			s.scanReads(arg)
		}
	}
}

func (s *constBodyScan) markArgs(args []syntax.Expr) {
	for _, arg := range args {
		s.markMention(arg)
	}
}
