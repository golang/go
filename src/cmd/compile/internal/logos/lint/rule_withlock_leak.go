// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmp"
)

var withlockLeak = Rule{
	ID:      "correctness::withlock_leak",
	Group:   Correctness,
	Default: Deny,
	Go:      true,
	Check:   checkWithlockLeak,
}

type leakClosure struct {
	node     syntax.Node
	body     syntax.Node
	captures bool
	escapes  bool
}

type leakReturn struct {
	stmt    *syntax.ReturnStmt
	closure *leakClosure
}

type leakCheck struct {
	pass       *Pass
	paramName  string
	paramObjs  []types2.Object
	locals     map[types2.Object]bool
	aliases    map[types2.Object]bool
	aliasKills map[types2.Object][]*syntax.AssignStmt
	body       syntax.Node
	closures   []*leakClosure
	returns    []leakReturn
}

func checkWithlockLeak(pass *Pass) {
	seen := make(map[*syntax.FuncDecl]bool)
	for _, f := range pass.Files {
		syntax.Inspect(f, func(n syntax.Node) bool {
			call, ok := n.(*syntax.CallExpr)
			if !ok || len(call.ArgList) != 1 {
				return true
			}
			fn := calleeFunc(pass, call)
			if fn == nil || fn.Name() != "WithLock" || fn.Pkg() == nil || fn.Pkg().Path() != logosSyncPath {
				return true
			}
			arg := syntax.Unparen(call.ArgList[0])
			name, objs := leakParam(pass, arg)
			if name == "" || len(objs) == 0 {
				return true
			}
			body := leakCallbackBody(pass, arg, seen)
			if body == nil {
				return true
			}
			c := &leakCheck{pass: pass, paramName: name, paramObjs: objs, locals: make(map[types2.Object]bool), aliases: make(map[types2.Object]bool), body: body}
			c.indexLocals(body)
			c.buildAliases(body)
			c.recordAliasKills(body)
			c.scanUnit(body, nil)
			c.markEscapes(body)
			for _, r := range c.returns {
				if r.closure == nil || r.closure.escapes {
					pass.Reportf(r.stmt, "WithLock argument must not return from the callback")
				}
			}
			return true
		})
	}
}

func leakCallbackBody(pass *Pass, arg syntax.Expr, seen map[*syntax.FuncDecl]bool) syntax.Node {
	switch a := arg.(type) {
	case *syntax.FuncLit:
		return a.Body
	case *syntax.LambdaExpr:
		return a.Body
	case *syntax.Name, *syntax.SelectorExpr:
		decl, _ := namedCallback(pass, arg)
		if decl == nil || decl.Body == nil || seen[decl] {
			return nil
		}
		seen[decl] = true
		return decl.Body
	}
	return nil
}

func namedCallback(pass *Pass, arg syntax.Expr) (*syntax.FuncDecl, *syntax.Field) {
	var fn *types2.Func
	switch a := arg.(type) {
	case *syntax.Name:
		fn, _ = pass.Info.Uses[a].(*types2.Func)
	case *syntax.SelectorExpr:
		fn, _ = pass.Info.Uses[a.Sel].(*types2.Func)
	}
	if fn == nil || fn.Pkg() == nil || fn.Pkg() != pass.Pkg {
		return nil, nil
	}
	for _, f := range pass.Files {
		for _, d := range f.DeclList {
			decl, ok := d.(*syntax.FuncDecl)
			if !ok || decl.Body == nil {
				continue
			}
			if obj, ok := pass.Info.Defs[decl.Name].(*types2.Func); !ok || obj != origin(fn) {
				continue
			}
			params := decl.Type.ParamList
			if len(params) == 0 || params[0].Name == nil {
				return nil, nil
			}
			return decl, params[0]
		}
	}
	return nil, nil
}

func leakParam(pass *Pass, arg syntax.Expr) (string, []types2.Object) {
	var name string
	var decls []*syntax.Name
	switch a := arg.(type) {
	case *syntax.FuncLit:
		if a.Type != nil {
			for _, field := range a.Type.ParamList {
				if field.Name != nil {
					decls = append(decls, field.Name)
				}
			}
		}
	case *syntax.LambdaExpr:
		decls = append(decls, a.Params...)
	case *syntax.Name, *syntax.SelectorExpr:
		if _, param := namedCallback(pass, arg); param != nil && param.Name != nil {
			decls = append(decls, param.Name)
		}
	}
	if len(decls) == 0 {
		return "", nil
	}
	name = decls[0].Value
	var objs []types2.Object
	add := func(o types2.Object) {
		if o == nil {
			return
		}
		for _, have := range objs {
			if have == o {
				return
			}
		}
		objs = append(objs, o)
	}
	if sig, ok := TypeOf(arg).(*types2.Signature); ok && sig.Params().Len() == 1 {
		add(sig.Params().At(0))
	}
	for _, d := range decls {
		add(pass.Info.Defs[d])
	}
	return name, objs
}

func (c *leakCheck) isParamObj(obj types2.Object) bool {
	if obj == nil {
		return false
	}
	for _, o := range c.paramObjs {
		if o == obj {
			return true
		}
	}
	return false
}

func (c *leakCheck) isParamUse(name *syntax.Name) bool {
	obj, ok := c.pass.Info.Uses[name]
	if !ok {
		return false
	}
	return c.isParamObj(obj)
}

func (c *leakCheck) isAliasUse(name *syntax.Name) bool {
	obj, ok := c.pass.Info.Uses[name]
	if !ok {
		return false
	}
	return c.aliases[obj]
}

func (c *leakCheck) indexLocals(body syntax.Node) {
	syntax.Inspect(body, func(n syntax.Node) bool {
		name, ok := n.(*syntax.Name)
		if !ok {
			return true
		}
		if obj, ok := c.pass.Info.Defs[name]; ok && !c.isParamObj(obj) {
			c.locals[obj] = true
		}
		return true
	})
}

func (c *leakCheck) buildAliases(body syntax.Node) {
	for grew := true; grew; {
		grew = false
		syntax.Inspect(body, func(n syntax.Node) bool {
			if x, ok := n.(*syntax.AssignStmt); ok && x.Rhs != nil && c.aliasFromAssign(x) {
				grew = true
			}
			return true
		})
	}
}

func (c *leakCheck) aliasFromAssign(x *syntax.AssignStmt) bool {
	lhs, rhs := exprList(x.Lhs), exprList(x.Rhs)
	grew := false
	for i, el := range lhs {
		if i >= len(rhs) {
			break
		}
		name, ok := syntax.Unparen(el).(*syntax.Name)
		if !ok || isBlank(name) {
			continue
		}
		obj := cmp.Or(c.pass.Info.Defs[name], c.pass.Info.Uses[name])
		if obj == nil || c.isParamObj(obj) || c.aliases[obj] {
			continue
		}
		if c.carriesPtr(rhs[i]) || c.isDerefCopy(rhs[i]) {
			c.aliases[obj] = true
			grew = true
		}
	}
	return grew
}
func (c *leakCheck) recordAliasKills(body syntax.Node) {
	c.aliasKills = make(map[types2.Object][]*syntax.AssignStmt)
	syntax.Inspect(body, func(n syntax.Node) bool {
		x, ok := n.(*syntax.AssignStmt)
		if !ok || x.Rhs == nil {
			return true
		}
		lhs, rhs := exprList(x.Lhs), exprList(x.Rhs)
		for i, el := range lhs {
			if i >= len(rhs) {
				break
			}
			name, ok := syntax.Unparen(el).(*syntax.Name)
			if !ok || isBlank(name) {
				continue
			}
			obj := cmp.Or(c.pass.Info.Defs[name], c.pass.Info.Uses[name])
			if obj == nil || !c.aliases[obj] {
				continue
			}
			if c.carriesPtr(rhs[i]) || c.isDerefCopy(rhs[i]) {
				continue
			}
			c.aliasKills[obj] = append(c.aliasKills[obj], x)
		}
		return true
	})
}

func (c *leakCheck) aliasKilled(rhs syntax.Expr, site *syntax.AssignStmt) bool {
	killed, active := false, false
	syntax.Inspect(rhs, func(n syntax.Node) bool {
		name, ok := n.(*syntax.Name)
		if !ok {
			return true
		}
		obj := c.pass.Info.Uses[name]
		if obj == nil {
			return true
		}
		if c.isParamObj(obj) {
			active = true
			return true
		}
		if !c.aliases[obj] {
			return true
		}
		if c.killedBefore(obj, site) {
			killed = true
		} else {
			active = true
		}
		return true
	})
	return killed && !active
}

func (c *leakCheck) killedBefore(obj types2.Object, site *syntax.AssignStmt) bool {
	for _, k := range c.aliasKills[obj] {
		if k == site || k.Pos().Cmp(site.Pos()) >= 0 {
			continue
		}
		if leakParent(c.body, k) == leakParent(c.body, site) {
			return true
		}
	}
	return false
}

func (c *leakCheck) scanUnit(body syntax.Node, cl *leakClosure) {
	syntax.Inspect(body, func(n syntax.Node) bool {
		switch x := n.(type) {
		case *syntax.FuncLit:
			nc := &leakClosure{node: x, body: x.Body}
			c.closures = append(c.closures, nc)
			c.scanUnit(x.Body, nc)
			return false
		case *syntax.LambdaExpr:
			nc := &leakClosure{node: x, body: x.Body}
			c.closures = append(c.closures, nc)
			c.scanUnit(x.Body, nc)
			return false
		case *syntax.ReturnStmt:
			if x.Results != nil && c.carriesPtr(x.Results) {
				c.returns = append(c.returns, leakReturn{stmt: x, closure: cl})
			}
		case *syntax.AssignStmt:
			if x.Rhs != nil && c.carriesPtr(x.Rhs) && c.lhsEscapes(x.Lhs) && !c.aliasKilled(x.Rhs, x) {
				c.pass.Reportf(x, "WithLock argument must not be stored outside the callback")
			}
		case *syntax.SendStmt:
			if c.carriesPtr(x.Value) {
				c.pass.Reportf(x, "WithLock argument must not be sent on a channel")
			}
		case *syntax.CallStmt:
			if x.Tok == syntax.Go {
				if call := callOf(x.Call); call != nil && c.spawnEscapes(call) {
					c.pass.Reportf(x, "WithLock argument must not be passed to a goroutine")
				}
			}
		}
		return true
	})
}

func (c *leakCheck) spawnEscapes(call *syntax.CallExpr) bool {
	for _, a := range call.ArgList {
		if c.carriesPtr(a) {
			return true
		}
	}
	switch fun := syntax.Unparen(call.Fun).(type) {
	case *syntax.FuncLit:
		return c.captures(fun.Body)
	case *syntax.LambdaExpr:
		return c.captures(fun.Body)
	case *syntax.SelectorExpr:
		return c.receiverEscapes(fun)
	}
	return false
}

func (c *leakCheck) receiverEscapes(sel *syntax.SelectorExpr) bool {
	s := c.pass.Info.Selections[sel]
	if s == nil || s.Kind() != types2.MethodVal {
		return false
	}
	x := syntax.Unparen(sel.X)
	if recv := methodRecv(s); recv == nil || isReferenceType(recv) {
		return c.rootedAtBorrowed(x) || c.carriesPtr(x)
	}
	return !isPointerType(TypeOf(x)) && c.carriesPtr(x)
}

func (c *leakCheck) rootedAtBorrowed(e syntax.Expr) bool {
	for {
		switch x := syntax.Unparen(e).(type) {
		case *syntax.Name:
			return c.isParamUse(x) || c.isAliasUse(x)
		case *syntax.SelectorExpr:
			e = x.X
		case *syntax.IndexExpr:
			e = x.X
		case *syntax.SliceExpr:
			e = x.X
		case *syntax.Operation:
			if x.Y != nil || (x.Op != syntax.Mul && x.Op != syntax.And) {
				return false
			}
			e = x.X
		default:
			return false
		}
	}
}

func methodRecv(s *types2.Selection) types2.Type {
	fn, ok := s.Obj().(*types2.Func)
	if !ok {
		return nil
	}
	if r := fn.Signature().Recv(); r != nil {
		return r.Type()
	}
	return nil
}

func isPointerType(t types2.Type) bool {
	if t == nil {
		return false
	}
	_, ok := types2.Unalias(t).Underlying().(*types2.Pointer)
	return ok
}

func (c *leakCheck) carriesPtr(e syntax.Expr) bool {
	found := false
	syntax.Inspect(e, func(n syntax.Node) bool {
		if found {
			return false
		}
		name, ok := n.(*syntax.Name)
		if !ok {
			return true
		}
		if !c.isParamUse(name) && !c.isAliasUse(name) {
			return true
		}
		if p := leakParent(e, name); p != nil {
			if op, ok := p.(*syntax.Operation); ok && op.Op == syntax.Mul && op.Y == nil {
				if operand, ok := op.X.(*syntax.Name); ok && operand == name {
					return true
				}
			}
			if sel, ok := p.(*syntax.SelectorExpr); ok {
				if operand, ok := sel.X.(*syntax.Name); ok && operand == name {
					if !addrTaken(e, sel) && !c.fieldLeaks(sel) {
						return true
					}
				}
			}
		}
		found = true
		return false
	})
	return found
}

func addrTaken(e syntax.Node, sel *syntax.SelectorExpr) bool {
	cur := syntax.Node(sel)
	for {
		p := leakParent(e, cur)
		if par, ok := p.(*syntax.ParenExpr); ok {
			cur = par
			continue
		}
		op, ok := p.(*syntax.Operation)
		return ok && op.Op == syntax.And && op.Y == nil && syntax.Unparen(op.X) == syntax.Node(sel)
	}
}

func (c *leakCheck) isDerefCopy(e syntax.Expr) bool {
	op, ok := syntax.Unparen(e).(*syntax.Operation)
	if !ok || op.Op != syntax.Mul || op.Y != nil {
		return false
	}
	name, ok := syntax.Unparen(op.X).(*syntax.Name)
	if !ok {
		return false
	}
	return c.isParamUse(name) || c.isAliasUse(name)
}

func (c *leakCheck) fieldLeaks(sel *syntax.SelectorExpr) bool {
	s := c.pass.Info.Selections[sel]
	if s == nil || s.Kind() != types2.FieldVal {
		return false
	}
	obj, ok := s.Obj().(*types2.Var)
	if !ok || obj.Type() == nil {
		return false
	}
	return isReferenceType(obj.Type())
}

func isReferenceType(t types2.Type) bool {
	switch types2.Unalias(t).Underlying().(type) {
	case *types2.Pointer, *types2.Slice, *types2.Map, *types2.Chan, *types2.Signature, *types2.Interface:
		return true
	}
	if b, ok := types2.Unalias(t).Underlying().(*types2.Basic); ok && b.Kind() == types2.UnsafePointer {
		return true
	}
	return false
}

func (c *leakCheck) captures(body syntax.Node) bool {
	captured := false
	syntax.Inspect(body, func(n syntax.Node) bool {
		if captured {
			return false
		}
		name, ok := n.(*syntax.Name)
		if !ok {
			return true
		}
		if c.isParamUse(name) || c.isAliasUse(name) {
			captured = true
			return false
		}
		return true
	})
	return captured
}

func (c *leakCheck) lhsEscapes(e syntax.Expr) bool {
	for _, el := range exprList(e) {
		if c.operandEscapes(syntax.Unparen(el)) {
			return true
		}
	}
	return false
}

func (c *leakCheck) operandEscapes(e syntax.Expr) bool {
	switch x := e.(type) {
	case *syntax.Name:
		if isBlank(x) {
			return false
		}
		obj := c.pass.Info.Uses[x]
		if obj == nil || c.isParamObj(obj) {
			return false
		}
		return !c.locals[obj]
	case *syntax.SelectorExpr, *syntax.IndexExpr:
		return true
	case *syntax.Operation:
		return c.operandEscapes(syntax.Unparen(x.X))
	case *syntax.ListExpr:
		for _, el := range x.ElemList {
			if c.lhsEscapes(el) {
				return true
			}
		}
	}
	return false
}

func (c *leakCheck) markEscapes(body syntax.Node) {
	for _, cl := range c.closures {
		cl.captures = c.captures(cl.body)
	}
	var escaping []syntax.Node
	syntax.Inspect(body, func(n syntax.Node) bool {
		switch x := n.(type) {
		case *syntax.ReturnStmt:
			if x.Results != nil {
				escaping = append(escaping, x.Results)
			}
		case *syntax.SendStmt:
			escaping = append(escaping, x.Value)
		case *syntax.CallStmt:
			if x.Tok == syntax.Go {
				if call := callOf(x.Call); call != nil {
					escaping = append(escaping, call.Fun)
					for _, a := range call.ArgList {
						escaping = append(escaping, a)
					}
				}
			}
		case *syntax.AssignStmt:
			if x.Rhs != nil && c.lhsEscapes(x.Lhs) {
				escaping = append(escaping, x.Rhs)
			}
		}
		return true
	})
	aliases := make(map[types2.Object]bool)
	syntax.Inspect(body, func(n syntax.Node) bool {
		x, ok := n.(*syntax.AssignStmt)
		if !ok || x.Rhs == nil {
			return true
		}
		for _, el := range exprList(x.Lhs) {
			name, ok := syntax.Unparen(el).(*syntax.Name)
			if !ok || isBlank(name) {
				continue
			}
			obj := c.pass.Info.Uses[name]
			if obj == nil || c.isParamObj(obj) || !c.locals[obj] {
				continue
			}
			for _, cl := range c.closures {
				if leakContains(x.Rhs, cl.node) {
					aliases[obj] = true
				}
			}
		}
		return true
	})
	for _, cl := range c.closures {
		if !cl.captures {
			continue
		}
		if leakContainsAny(escaping, cl.node) {
			cl.escapes = true
			continue
		}
		for obj := range aliases {
			if leakUses(c.pass, escaping, obj) {
				cl.escapes = true
				break
			}
		}
	}
}

func leakUses(pass *Pass, escaping []syntax.Node, obj types2.Object) bool {
	for _, r := range escaping {
		used := false
		syntax.Inspect(r, func(n syntax.Node) bool {
			if used {
				return false
			}
			name, ok := n.(*syntax.Name)
			if !ok {
				return true
			}
			if pass.Info.Uses[name] == obj {
				used = true
				return false
			}
			return true
		})
		if used {
			return true
		}
	}
	return false
}

func leakContains(root, target syntax.Node) bool {
	found := false
	syntax.Inspect(root, func(n syntax.Node) bool {
		if n == target {
			found = true
			return false
		}
		return !found
	})
	return found
}

func leakContainsAny(roots []syntax.Node, target syntax.Node) bool {
	for _, r := range roots {
		if leakContains(r, target) {
			return true
		}
	}
	return false
}

type leakParentFinder struct {
	target syntax.Node
	parent syntax.Node
	stack  []syntax.Node
}

func (v *leakParentFinder) Visit(n syntax.Node) syntax.Visitor {
	if v.parent != nil {
		return nil
	}
	if n == nil {
		v.stack = v.stack[:len(v.stack)-1]
		return v
	}
	if n == v.target {
		if len(v.stack) > 0 {
			v.parent = v.stack[len(v.stack)-1]
		}
		return nil
	}
	v.stack = append(v.stack, n)
	return v
}

func leakParent(root, target syntax.Node) syntax.Node {
	v := &leakParentFinder{target: target}
	syntax.Walk(root, v)
	return v.parent
}
