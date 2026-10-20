// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"slices"
	"strconv"

	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// This file carries the M5-S1 const facts (SPEC Sec 2.1, T-201 and
// Sec 6.1, T-Sec6). EmitConstFacts records one const line per const
// view position in the exported declarations of pkg, plus
// ReceiverMutated and ParamMutated flags computed by a syntax pass
// over each function body: a parameter counts as mutated when the body
// writes through it, appends to it, passes it (or a value derived
// from it) to a mutating parameter, or stores it anywhere outside the
// frame (globals, fields, returns, sends, closures, addresses). The
// parent wires the call into lower.go beside EmitOptionFacts; this
// file stays independent of the lowering passes.

func constAsView(t types2.Type) *types2.ConstView {
	if t == nil {
		return nil
	}
	if c, ok := types2.Unalias(t).(*types2.ConstView); ok {
		return c
	}
	if n, ok := types2.Unalias(t).(*types2.Named); ok {
		if u := n.Underlying(); u != nil {
			if c, ok := u.(*types2.ConstView); ok {
				return c
			}
		}
	}
	return nil
}

// EmitConstFacts adds const view positions and receiver/parameter
// mutation flags for pkg to f (allocating it when nil) and returns f.
// Objects are func names, type names, and T.M or (*T).M methods;
// paths are R<i>, P<i>, F<name>, E, K, M<name> steps, mirroring the opt
// records. The mutation mask sets bit i for Pi, formatted as
// normalized lowercase hex. Generic functions are skipped: their
// bodies check per instantiation.
func EmitConstFacts(pkg *types2.Package, info *types2.Info, files []*syntax.File, f *facts.Facts) *facts.Facts {
	if pkg == nil || pkg.Scope() == nil {
		return f
	}
	if f == nil {
		f = &facts.Facts{}
	}
	emitConstDecls(pkg, f)
	emitConstMutations(pkg, info, files, f)
	return f
}

func emitConstDecls(pkg *types2.Package, f *facts.Facts) {
	emit := func(object, path string) {
		for _, c := range f.Consts {
			if c.Object == object && c.Path == path {
				return
			}
		}
		f.Consts = append(f.Consts, facts.PathFact{Object: object, Path: path})
	}
	scope := pkg.Scope()
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
			if tp := sig.TypeParams(); tp != nil && tp.Len() > 0 {
				continue
			}
			emitFuncConsts(name, sig, emit)
		case *types2.TypeName:
			if !obj.IsAlias() {
				emitTypeConsts(name, obj.Type(), emit)
			}
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
			if tp := sig.TypeParams(); tp != nil && tp.Len() > 0 {
				continue
			}
			emitFuncConsts(methodObject(name, sig.Recv(), m.Name()), sig, emit)
		}
	}
}

func emitFuncConsts(object string, sig *types2.Signature, emit func(string, string)) {
	if params := sig.Params(); params != nil {
		for i := range params.Len() {
			v := params.At(i)
			if v == nil {
				continue
			}
			walkConstType(v.Type(), []string{fmt.Sprintf("P%d", i)}, object, emit, make(map[types2.Type]bool))
		}
	}
	if results := sig.Results(); results != nil {
		for i := range results.Len() {
			v := results.At(i)
			if v == nil {
				continue
			}
			walkConstType(v.Type(), []string{fmt.Sprintf("R%d", i)}, object, emit, make(map[types2.Type]bool))
		}
	}
}

func emitTypeConsts(name string, t types2.Type, emit func(string, string)) {
	u, ok := types2.Unalias(t).(*types2.Named)
	if !ok {
		return
	}
	if tp := u.TypeParams(); tp != nil && tp.Len() > 0 {
		return
	}
	switch u := u.Underlying().(type) {
	case *types2.Struct:
		for i := range u.NumFields() {
			f := u.Field(i)
			if f == nil || !f.Exported() {
				continue
			}
			walkConstType(f.Type(), []string{"F" + f.Name()}, name, emit, make(map[types2.Type]bool))
		}
	case *types2.Interface:
		for i := range u.NumExplicitMethods() {
			m := u.ExplicitMethod(i)
			if m == nil || !m.Exported() {
				continue
			}
			sig, ok := m.Type().(*types2.Signature)
			if !ok {
				continue
			}
			base := []string{"M" + m.Name()}
			if params := sig.Params(); params != nil {
				for j := range params.Len() {
					v := params.At(j)
					if v == nil {
						continue
					}
					walkConstType(v.Type(), append(slices.Clone(base), fmt.Sprintf("P%d", j)), name, emit, make(map[types2.Type]bool))
				}
			}
			if results := sig.Results(); results != nil {
				for j := range results.Len() {
					v := results.At(j)
					if v == nil {
						continue
					}
					walkConstType(v.Type(), append(slices.Clone(base), fmt.Sprintf("R%d", j)), name, emit, make(map[types2.Type]bool))
				}
			}
		}
	}
}

func walkConstType(t types2.Type, path []string, object string, emit func(string, string), seen map[types2.Type]bool) {
	if t == nil {
		return
	}
	if c := constAsView(t); c != nil {
		emit(object, joinPath(path))
		walkConstType(c.Elem, append(slices.Clone(path), "E"), object, emit, seen)
		return
	}
	u := types2.Unalias(t)
	if seen[u] {
		return
	}
	seen[u] = true
	switch u := u.(type) {
	case *types2.Slice:
		walkConstType(u.Elem(), append(slices.Clone(path), "E"), object, emit, seen)
	case *types2.Array:
		walkConstType(u.Elem(), append(slices.Clone(path), "E"), object, emit, seen)
	case *types2.Map:
		walkConstType(u.Key(), append(slices.Clone(path), "K"), object, emit, seen)
		walkConstType(u.Elem(), append(slices.Clone(path), "E"), object, emit, seen)
	case *types2.Pointer:
		walkConstType(u.Elem(), path, object, emit, seen)
	case *types2.Chan:
		walkConstType(u.Elem(), path, object, emit, seen)
	case *types2.Signature:
		if params := u.Params(); params != nil {
			for i := range params.Len() {
				v := params.At(i)
				if v == nil {
					continue
				}
				walkConstType(v.Type(), append(slices.Clone(path), fmt.Sprintf("P%d", i)), object, emit, seen)
			}
		}
		if results := u.Results(); results != nil {
			for i := range results.Len() {
				v := results.At(i)
				if v == nil {
					continue
				}
				walkConstType(v.Type(), append(slices.Clone(path), fmt.Sprintf("R%d", i)), object, emit, seen)
			}
		}
	case *types2.Struct:
		for i := range u.NumFields() {
			f := u.Field(i)
			if f == nil || !f.Exported() {
				continue
			}
			walkConstType(f.Type(), append(slices.Clone(path), "F"+f.Name()), object, emit, seen)
		}
	case *types2.Interface:
		for i := range u.NumExplicitMethods() {
			m := u.ExplicitMethod(i)
			if m == nil || !m.Exported() {
				continue
			}
			sig, ok := m.Type().(*types2.Signature)
			if !ok {
				continue
			}
			mbase := append(slices.Clone(path), "M"+m.Name())
			if params := sig.Params(); params != nil {
				for j := range params.Len() {
					v := params.At(j)
					if v == nil {
						continue
					}
					walkConstType(v.Type(), append(slices.Clone(mbase), fmt.Sprintf("P%d", j)), object, emit, seen)
				}
			}
			if results := sig.Results(); results != nil {
				for j := range results.Len() {
					v := results.At(j)
					if v == nil {
						continue
					}
					walkConstType(v.Type(), append(slices.Clone(mbase), fmt.Sprintf("R%d", j)), object, emit, seen)
				}
			}
		}
	}
}

// constFuncDecl pairs a declared function with its syntax and type.
type constFuncDecl struct {
	obj  *types2.Func
	decl *syntax.FuncDecl
	sig  *types2.Signature
	name string
}

// emitConstMutations computes ReceiverMutated and ParamMutated flags by
// a syntax pass over every package-level function body and merges them
// into f. Masks propagate between same-package callees to a fixpoint;
// imported callees consult the recorded facts.
func emitConstMutations(pkg *types2.Package, info *types2.Info, files []*syntax.File, f *facts.Facts) {
	decls := collectConstFuncDecls(pkg, info, files)
	if len(decls) == 0 {
		return
	}
	recvMut := make(map[*types2.Func]bool, len(decls))
	paramMask := make(map[*types2.Func]uint64, len(decls))
	for _, d := range decls {
		paramMask[d.obj] = 0
	}
	for range len(decls) + 1 {
		changed := false
		for _, d := range decls {
			scan := &constBodyScan{pkg: pkg, info: info, recvMut: recvMut, paramMask: paramMask}
			rm, mask := scan.funcMutations(d)
			if rm && !recvMut[d.obj] {
				recvMut[d.obj] = true
				changed = true
			}
			if mask != paramMask[d.obj] {
				paramMask[d.obj] = mask
				changed = true
			}
		}
		if !changed {
			break
		}
	}
	for _, d := range decls {
		if !recvMut[d.obj] && paramMask[d.obj] == 0 {
			continue
		}
		entry := -1
		for i := range f.Fns {
			if f.Fns[i].Object == d.name {
				entry = i
				break
			}
		}
		if entry < 0 {
			f.Fns = append(f.Fns, facts.FnFact{Object: d.name})
			entry = len(f.Fns) - 1
		}
		fn := &f.Fns[entry]
		fn.ReceiverMutated = fn.ReceiverMutated || recvMut[d.obj]
		fn.ParamMutated = orHexMask(fn.ParamMutated, paramMask[d.obj])
	}
}

func orHexMask(have string, add uint64) string {
	var cur uint64
	if have != "" {
		if v, err := strconv.ParseUint(have, 16, 64); err == nil {
			cur = v
		}
	}
	cur |= add
	if cur == 0 {
		return ""
	}
	return strconv.FormatUint(cur, 16)
}

// collectConstFuncDecls matches every package-level function body with
// its declared object. Generic functions are skipped.
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

// constRecvBase strips pointer and const view layers from a receiver
// type to its named base.
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

// constFuncName renders the facts object name: the func name, or
// T.M / (*T).M for methods.
func constFuncName(pkg *types2.Package, obj *types2.Func, sig *types2.Signature) string {
	if sig.Recv() == nil {
		return obj.Name()
	}
	if named, ok := constRecvBase(sig.Recv().Type()).(*types2.Named); ok && named.Obj().Pkg() == pkg {
		return methodObject(named.Obj().Name(), sig.Recv(), obj.Name())
	}
	return obj.Name()
}

// constBodyScan holds the state for scanning one function body.
// tainted maps an object derived from a parameter to its origin
// parameter; mut records confirmed-mutated origin parameters.
type constBodyScan struct {
	pkg       *types2.Package
	info      *types2.Info
	recvMut   map[*types2.Func]bool
	paramMask map[*types2.Func]uint64
	tainted   map[types2.Object]*types2.Var
	mut       map[*types2.Var]bool
}

// funcMutations returns the receiver-mutated flag and the parameter
// mutation mask for one function.
func (s *constBodyScan) funcMutations(d constFuncDecl) (bool, uint64) {
	s.tainted = make(map[types2.Object]*types2.Var)
	s.mut = make(map[*types2.Var]bool)
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
	s.block(d.decl.Body.List)
	var mask uint64
	bit := 0
	for _, p := range params {
		if p == recv {
			continue
		}
		if bit < 64 && s.mut[p] {
			mask |= 1 << uint(bit)
		}
		bit++
	}
	return recv != nil && s.mut[recv], mask
}

// resolve maps a name to its declared object through definitions first,
// then uses.
func (s *constBodyScan) resolve(name *syntax.Name) types2.Object {
	if name == nil || s.info == nil {
		return nil
	}
	if obj, ok := s.info.Defs[name]; ok && obj != nil {
		return obj
	}
	return s.info.Uses[name]
}

// isPackageVar reports whether obj is a package-level variable.
func (s *constBodyScan) isPackageVar(obj types2.Object) bool {
	v, ok := obj.(*types2.Var)
	if !ok || v == nil || s.pkg == nil {
		return false
	}
	return s.pkg.Scope().Lookup(v.Name()) == types2.Object(v)
}

// originOf returns the origin parameter of a tainted object, if any.
func (s *constBodyScan) originOf(obj types2.Object) *types2.Var {
	if obj == nil {
		return nil
	}
	return s.tainted[obj]
}

// mark records the origin parameter of obj as mutated.
func (s *constBodyScan) mark(obj types2.Object) {
	if o := s.originOf(obj); o != nil {
		s.mut[o] = true
	}
}

// taintedMention returns the origin parameter mentioned in e, if any.
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

// block scans a statement list.
func (s *constBodyScan) block(list []syntax.Stmt) {
	for _, st := range list {
		s.stmt(st)
	}
}

// stmt scans one statement, recursing into nested bodies.
func (s *constBodyScan) stmt(st syntax.Stmt) {
	switch st := st.(type) {
	case *syntax.AssignStmt:
		s.assign(st)
	case *syntax.ReturnStmt:
		if st.Results != nil {
			s.markMention(st.Results)
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
		if st.Cond != nil {
			s.scanReads(st.Cond)
		}
		s.blockStmt(st.Then)
		if st.Else != nil {
			s.stmt(st.Else)
		}
	case *syntax.ForStmt:
		if st.Init != nil {
			s.scanReads(st.Init)
		}
		if st.Cond != nil {
			s.scanReads(st.Cond)
		}
		if st.Post != nil {
			s.scanReads(st.Post)
		}
		s.blockStmt(st.Body)
	case *syntax.SwitchStmt:
		if st.Init != nil {
			s.scanReads(st.Init)
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
				s.scanReads(c.Comm)
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
		s.scanReads(st)
	}
}

// blockStmt scans a block-shaped child: a block or a statement.
func (s *constBodyScan) blockStmt(n syntax.Node) {
	switch n := n.(type) {
	case *syntax.BlockStmt:
		s.block(n.List)
	case syntax.Stmt:
		s.stmt(n)
	}
}

// assign scans an assignment: writes through tainted values, global
// stores of tainted values, and taint propagation into locals.
func (s *constBodyScan) assign(st *syntax.AssignStmt) {
	if st.Rhs == nil {
		s.markMention(st.Lhs)
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
					s.tainted[obj] = o
				} else if lhs.Value != "_" {
					s.markMention(st.Rhs)
				}
				return
			}
			if obj != nil {
				delete(s.tainted, obj)
			}
			return
		}
		s.markMention(st.Lhs)
		if s.mentionsPackage(st.Lhs) {
			s.markMention(st.Rhs)
		} else {
			s.scanReads(st.Rhs)
		}
		return
	}
	s.markMention(st.Lhs)
}

// mentionsPackage reports whether e mentions a package-level variable.
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

// varDecl propagates taint into var declarations with initializers.
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
		} else {
			delete(s.tainted, obj)
		}
	}
}

// suspended scans go and defer statements: a go statement escapes its
// arguments into a new frame, while defer runs in the frame.
func (s *constBodyScan) suspended(st *syntax.CallStmt) {
	if st.Tok == syntax.Go {
		s.markMention(st.Call)
		return
	}
	s.scanReads(st.Call)
}

// markMention marks the origin of every tainted value mentioned in e:
// the whole expression escapes, so even pure-position arguments mark.
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
		return true
	})
}

// scanReads analyzes e for call effects and captured closures,
// ignoring bare reads.
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
		return true
	})
}

// markClosure marks every tainted value captured by a function
// literal body.
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

// callInFrame scans a call whose result stays in the frame; only
// mutating positions mark.
func (s *constBodyScan) callInFrame(call *syntax.CallExpr) {
	s.doCall(call, false)
}

// callEscaping scans a call whose result escapes; even pure-position
// arguments mark, since the result may alias them.
func (s *constBodyScan) callEscaping(call *syntax.CallExpr) {
	s.doCall(call, true)
}

// doCall scans a call: arguments against the callee parameter facts,
// the receiver against the receiver fact, and builtins by position.
func (s *constBodyScan) doCall(call *syntax.CallExpr, escaping bool) {
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
		}
		s.markArgs(call.ArgList)
		return
	}
	if sel, ok := call.Fun.(*syntax.SelectorExpr); ok {
		if s.info != nil {
			if choice, ok := s.info.Selections[sel]; ok && choice != nil {
				if fn, ok := choice.Obj().(*types2.Func); ok {
					s.doFuncCall(fn, sel.X, call.ArgList, escaping)
					return
				}
			}
		}
		s.markArgs(call.ArgList)
		return
	}
	s.markArgs(call.ArgList)
}

// funcCall checks arguments against a callee: same-package functions
// use the fixpoint masks, imported ones the recorded facts, and
// bodies without facts fail closed.
func (s *constBodyScan) doFuncCall(fn *types2.Func, recv syntax.Expr, args []syntax.Expr, escaping bool) {
	sig, ok := fn.Type().(*types2.Signature)
	if !ok {
		s.markArgs(args)
		return
	}
	mutRecv, mask := s.calleeFacts(fn)
	if recv != nil {
		if mutRecv || escaping {
			s.markMention(recv)
		} else {
			s.scanReads(recv)
		}
	}
	if ps := sig.Params(); ps != nil {
		for i, arg := range args {
			if i >= ps.Len() {
				s.scanReads(arg)
				break
			}
			if mask&(1<<uint(i)) != 0 || escaping {
				s.markMention(arg)
			} else {
				s.scanReads(arg)
			}
		}
	} else {
		s.markArgs(args)
	}
}

// calleeFacts returns the receiver flag and parameter mask for a
// callee: the fixpoint state for the current package, recorded facts
// for imports, and fail-closed for unknown bodies. A universe method
// such as error.Error has no package and no recorded facts.
func (s *constBodyScan) calleeFacts(fn *types2.Func) (bool, uint64) {
	if fn.Pkg() == nil {
		return true, ^uint64(0)
	}
	if fn.Pkg() == s.pkg {
		if mask, ok := s.paramMask[fn]; ok {
			return s.recvMut[fn], mask
		}
		return true, ^uint64(0)
	}
	if f, ok := facts.Lookup(fn.Pkg().Path()); ok && f != nil {
		for i := range f.Fns {
			entry := &f.Fns[i]
			if entry.Object != fn.Name() && entry.Object != constMethodName(fn) {
				continue
			}
			if entry.MayYieldOnly() {
				continue
			}
			var mask uint64
			if entry.ParamMutated != "" {
				if v, err := strconv.ParseUint(entry.ParamMutated, 16, 64); err == nil {
					mask = v
				} else {
					mask = ^uint64(0)
				}
			}
			return entry.ReceiverMutated, mask
		}
	}
	return true, ^uint64(0)
}

// constMethodName renders T.M or (*T).M for a method.
func constMethodName(fn *types2.Func) string {
	sig, ok := fn.Type().(*types2.Signature)
	if !ok || sig.Recv() == nil {
		return fn.Name()
	}
	if named, ok := constRecvBase(sig.Recv().Type()).(*types2.Named); ok {
		return methodObject(named.Obj().Name(), sig.Recv(), fn.Name())
	}
	return fn.Name()
}

// builtinCall scans built-in calls by position: append and copy mutate
// their destination, delete and clear mutate their map, panic escapes
// its argument, and every other built-in reads.
func (s *constBodyScan) builtinCall(name string, call *syntax.CallExpr) {
	switch name {
	case "append", "copy":
		if len(call.ArgList) > 0 {
			s.markMention(call.ArgList[0])
		}
	case "delete", "clear", "panic":
		for _, arg := range call.ArgList {
			s.markMention(arg)
		}
	}
}

// markArgs marks tainted mentions across call arguments.
func (s *constBodyScan) markArgs(args []syntax.Expr) {
	for _, arg := range args {
		s.markMention(arg)
	}
}
