package syntax

import "strconv"

// RewriteTuplePatternSwitch lowers tuple pattern matching sugar in switch statements
// before the first types2 pass.
//
// Recognize:
//
//	switch (a, b) { case (p, q): ...; default: ... }
//
// and desugar to:
//
//	{
//	  _t0 := a
//	  _t1 := b
//	  _matched := false
//	  _lbl: switch _t0 {
//	  case p:
//	    switch _t1 {
//	    case q:
//	      _matched = true
//	      ...
//	      break _lbl
//	    }
//	  }
//	  // default triggers when no tuple case matched (including first-match-but-second-miss).
//	  switch _matched {
//	  case false:
//	    ...
//	  }
//	}
//
// Notes/constraints (current implementation):
//   - Only 2-tuples are supported: (x, y).
//   - All non-default cases must be tuple patterns of the form (p, q).
//   - For correctness we require the first component patterns to be disjoint across groups:
//     if multiple cases use the same enum-variant constructor in the first slot but spell it
//     differently (e.g. Value.Int(a) vs Value.Int(_)), we conservatively skip rewriting.
func RewriteTuplePatternSwitch(file *File) {
	if file == nil {
		return
	}
	r := &tupleSwitchPreRewriter{}
	r.rewriteFile(file)
}

type tupleSwitchPreRewriter struct {
	tmpCounter int
}

func (r *tupleSwitchPreRewriter) gensym(pos Pos, prefix string) *Name {
	r.tmpCounter++
	return NewName(pos, prefix+strconv.Itoa(r.tmpCounter))
}

func (r *tupleSwitchPreRewriter) rewriteFile(file *File) {
	for _, decl := range file.DeclList {
		r.rewriteDecl(decl)
	}
}

func (r *tupleSwitchPreRewriter) rewriteDecl(d Decl) {
	switch d := d.(type) {
	case *FuncDecl:
		if d.Body != nil {
			d.Body = r.rewriteBlock(d.Body)
		}
		// Recurse into default parameter expressions etc. (best-effort).
		if d.Type != nil {
			r.rewriteFuncType(d.Type)
		}
		if d.Recv != nil {
			r.rewriteField(d.Recv)
		}
		for _, f := range d.TParamList {
			r.rewriteField(f)
		}
	case *VarDecl:
		if d.Type != nil {
			d.Type = r.rewriteExpr(d.Type)
		}
		if d.Values != nil {
			d.Values = r.rewriteExpr(d.Values)
		}
	case *ConstDecl:
		if d.Type != nil {
			d.Type = r.rewriteExpr(d.Type)
		}
		if d.Values != nil {
			d.Values = r.rewriteExpr(d.Values)
		}
	case *TypeDecl:
		if d.Type != nil {
			d.Type = r.rewriteExpr(d.Type)
		}
		for _, f := range d.TParamList {
			r.rewriteField(f)
		}
	case *ImportDecl:
		// nothing
	}
}

func (r *tupleSwitchPreRewriter) rewriteField(f *Field) {
	if f == nil {
		return
	}
	if f.Type != nil {
		f.Type = r.rewriteExpr(f.Type)
	}
	if f.DefaultValue != nil {
		f.DefaultValue = r.rewriteExpr(f.DefaultValue)
	}
}

func (r *tupleSwitchPreRewriter) rewriteFuncType(t *FuncType) {
	if t == nil {
		return
	}
	for _, f := range t.ParamList {
		r.rewriteField(f)
	}
	for _, f := range t.ResultList {
		r.rewriteField(f)
	}
}

func (r *tupleSwitchPreRewriter) rewriteBlock(b *BlockStmt) *BlockStmt {
	if b == nil {
		return b
	}
	for i, s := range b.List {
		b.List[i] = r.rewriteStmt(s)
	}
	return b
}

func (r *tupleSwitchPreRewriter) rewriteStmt(s Stmt) Stmt {
	switch s := s.(type) {
	case *EmptyStmt:
		return s
	case *LabeledStmt:
		s.Stmt = r.rewriteStmt(s.Stmt)
		return s
	case *BlockStmt:
		return r.rewriteBlock(s)
	case *ExprStmt:
		s.X = r.rewriteExpr(s.X)
		return s
	case *SendStmt:
		s.Chan = r.rewriteExpr(s.Chan)
		s.Value = r.rewriteExpr(s.Value)
		return s
	case *DeclStmt:
		for _, d := range s.DeclList {
			r.rewriteDecl(d)
		}
		return s
	case *AssignStmt:
		s.Lhs = r.rewriteExpr(s.Lhs)
		if s.Rhs != nil {
			s.Rhs = r.rewriteExpr(s.Rhs)
		}
		return s
	case *BranchStmt:
		return s
	case *CallStmt:
		s.Call = r.rewriteExpr(s.Call)
		if s.DeferAt != nil {
			s.DeferAt = r.rewriteExpr(s.DeferAt)
		}
		return s
	case *ReturnStmt:
		if s.Results != nil {
			s.Results = r.rewriteExpr(s.Results)
		}
		return s
	case *IfStmt:
		if s.Init != nil {
			s.Init = r.rewriteSimpleStmt(s.Init)
		}
		if s.Cond != nil {
			s.Cond = r.rewriteExpr(s.Cond)
		}
		s.Then = r.rewriteBlock(s.Then)
		if s.Else != nil {
			s.Else = r.rewriteStmt(s.Else)
		}
		return s
	case *ForStmt:
		if s.Init != nil {
			s.Init = r.rewriteSimpleStmt(s.Init)
		}
		if s.Cond != nil {
			s.Cond = r.rewriteExpr(s.Cond)
		}
		if s.Post != nil {
			s.Post = r.rewriteSimpleStmt(s.Post)
		}
		s.Body = r.rewriteBlock(s.Body)
		return s
	case *SwitchStmt:
		if s.Init != nil {
			s.Init = r.rewriteSimpleStmt(s.Init)
		}
		if s.Tag != nil {
			s.Tag = r.rewriteExpr(s.Tag)
		}
		for _, cc := range s.Body {
			r.rewriteCaseClause(cc)
		}
		if ns := r.desugarTupleSwitch(s); ns != nil {
			return ns
		}
		return s
	case *SelectStmt:
		for _, cc := range s.Body {
			r.rewriteCommClause(cc)
		}
		return s
	default:
		return s
	}
}

func (r *tupleSwitchPreRewriter) rewriteSimpleStmt(s SimpleStmt) SimpleStmt {
	switch s := s.(type) {
	case *EmptyStmt:
		return s
	case *ExprStmt:
		s.X = r.rewriteExpr(s.X)
		return s
	case *SendStmt:
		s.Chan = r.rewriteExpr(s.Chan)
		s.Value = r.rewriteExpr(s.Value)
		return s
	case *AssignStmt:
		s.Lhs = r.rewriteExpr(s.Lhs)
		if s.Rhs != nil {
			s.Rhs = r.rewriteExpr(s.Rhs)
		}
		return s
	case *RangeClause:
		if s.Lhs != nil {
			s.Lhs = r.rewriteExpr(s.Lhs)
		}
		s.X = r.rewriteExpr(s.X)
		return s
	default:
		return s
	}
}

func (r *tupleSwitchPreRewriter) rewriteCaseClause(cc *CaseClause) {
	if cc == nil {
		return
	}
	if cc.Cases != nil {
		cc.Cases = r.rewriteExpr(cc.Cases)
	}
	for i, s := range cc.Body {
		cc.Body[i] = r.rewriteStmt(s)
	}
}

func (r *tupleSwitchPreRewriter) rewriteCommClause(cc *CommClause) {
	if cc == nil {
		return
	}
	if cc.Comm != nil {
		cc.Comm = r.rewriteSimpleStmt(cc.Comm)
	}
	for i, s := range cc.Body {
		cc.Body[i] = r.rewriteStmt(s)
	}
}

func (r *tupleSwitchPreRewriter) rewriteExpr(e Expr) Expr {
	switch e := e.(type) {
	case nil:
		return nil
	case *Name, *BasicLit, *BadExpr:
		return e
	case *ParenExpr:
		e.X = r.rewriteExpr(e.X)
		return e
	case *ListExpr:
		for i, x := range e.ElemList {
			e.ElemList[i] = r.rewriteExpr(x)
		}
		return e
	case *Operation:
		e.X = r.rewriteExpr(e.X)
		if e.Y != nil {
			e.Y = r.rewriteExpr(e.Y)
		}
		return e
	case *SelectorExpr:
		e.X = r.rewriteExpr(e.X)
		return e
	case *OptionalChainExpr:
		e.X = r.rewriteExpr(e.X)
		return e
	case *IndexExpr:
		e.X = r.rewriteExpr(e.X)
		e.Index = r.rewriteExpr(e.Index)
		return e
	case *SliceExpr:
		e.X = r.rewriteExpr(e.X)
		for i := range e.Index {
			if e.Index[i] != nil {
				e.Index[i] = r.rewriteExpr(e.Index[i])
			}
		}
		return e
	case *CallExpr:
		e.Fun = r.rewriteExpr(e.Fun)
		for i, a := range e.ArgList {
			e.ArgList[i] = r.rewriteExpr(a)
		}
		return e
	case *CompositeLit:
		if e.Type != nil {
			e.Type = r.rewriteExpr(e.Type)
		}
		for i, el := range e.ElemList {
			e.ElemList[i] = r.rewriteExpr(el)
		}
		return e
	case *KeyValueExpr:
		e.Key = r.rewriteExpr(e.Key)
		e.Value = r.rewriteExpr(e.Value)
		return e
	case *FuncLit:
		if e.Type != nil {
			r.rewriteFuncType(e.Type)
		}
		e.Body = r.rewriteBlock(e.Body)
		return e
	case *TernaryExpr:
		e.Cond = r.rewriteExpr(e.Cond)
		e.X = r.rewriteExpr(e.X)
		if e.Y != nil {
			e.Y = r.rewriteExpr(e.Y)
		}
		return e
	case *CoalesceExpr:
		e.X = r.rewriteExpr(e.X)
		e.Y = r.rewriteExpr(e.Y)
		return e
	case *AssertExpr:
		e.X = r.rewriteExpr(e.X)
		e.Type = r.rewriteExpr(e.Type)
		return e
	case *TypeSwitchGuard:
		e.X = r.rewriteExpr(e.X)
		return e
	// Types (best-effort, to reach FuncLits embedded in type expressions, etc.)
	case *ArrayType:
		if e.Len != nil {
			e.Len = r.rewriteExpr(e.Len)
		}
		e.Elem = r.rewriteExpr(e.Elem)
		return e
	case *SliceType:
		e.Elem = r.rewriteExpr(e.Elem)
		return e
	case *DotsType:
		e.Elem = r.rewriteExpr(e.Elem)
		return e
	case *StructType:
		for _, f := range e.FieldList {
			r.rewriteField(f)
		}
		return e
	case *InterfaceType:
		for _, f := range e.MethodList {
			r.rewriteField(f)
		}
		return e
	case *MapType:
		e.Key = r.rewriteExpr(e.Key)
		e.Value = r.rewriteExpr(e.Value)
		return e
	case *ChanType:
		e.Elem = r.rewriteExpr(e.Elem)
		return e
	case *EnumType:
		for _, v := range e.VariantList {
			if v != nil && v.Type != nil {
				v.Type = r.rewriteExpr(v.Type)
			}
			if v != nil && v.Value != nil {
				v.Value = r.rewriteExpr(v.Value)
			}
		}
		return e
	case *FuncType:
		r.rewriteFuncType(e)
		return e
	default:
		return e
	}
}

func (r *tupleSwitchPreRewriter) desugarTupleSwitch(s *SwitchStmt) Stmt {
	if s == nil || s.Tag == nil {
		return nil
	}
	tagElems, ok := unpackTuple(s.Tag)
	if !ok {
		return nil
	}
	if len(tagElems) < 2 {
		return nil
	}

	// Validate and collect tuple cases.
	type tupleCase struct {
		pos  Pos
		pats []Expr
		body []Stmt
	}

	var cases []tupleCase
	var defaultBody []Stmt
	for _, cc := range s.Body {
		if cc == nil {
			return nil
		}
		if cc.Cases == nil {
			// default
			defaultBody = cc.Body
			continue
		}
		// Only support a single tuple pattern per clause: case (p, q):
		if _, isCaseList := cc.Cases.(*ListExpr); isCaseList {
			return nil
		}
		pats, ok := unpackTuple(cc.Cases)
		if !ok {
			return nil
		}
		if len(pats) != len(tagElems) {
			return nil
		}
		// Only support enum-variant patterns with Name-only payload patterns at every slot.
		// This enables safe lowering: bind payload to compiler temps, then re-bind user names
		// inside the leaf case body.
		for _, p := range pats {
			if !isVariantPatternNameArgs(p) {
				return nil
			}
		}
		cases = append(cases, tupleCase{
			pos:  cc.Pos(),
			pats: pats,
			body: cc.Body,
		})
	}
	if len(cases) == 0 {
		return nil
	}

	pos := s.Pos()

	// Build temps and matched flag.
	temps := make([]*Name, len(tagElems))
	for i := range temps {
		temps[i] = r.gensym(pos, "_tuple"+strconv.Itoa(i)+"_")
	}

	defAssign := func(name *Name, rhs Expr) *AssignStmt {
		as := new(AssignStmt)
		as.pos = pos
		as.Op = Def
		as.Lhs = NewName(pos, name.Value)
		as.Rhs = rhs
		return as
	}

	initStmts := make([]Stmt, 0, len(tagElems)+1)
	for i := range tagElems {
		initStmts = append(initStmts, defAssign(temps[i], tagElems[i]))
	}

	var defaultLabel *Name
	var endLabel *Name
	if len(defaultBody) > 0 {
		defaultLabel = r.gensym(pos, "_tuple_default_")
		endLabel = r.gensym(pos, "_tuple_end_")
	}

	gotoLabel := func(pos Pos, lab *Name) *BranchStmt {
		br := new(BranchStmt)
		br.pos = pos
		br.Tok = Goto
		br.Label = NewName(pos, lab.Value)
		return br
	}

	// Build nested switches:
	// - At each level, match only the enum variant and bind payload to compiler temps.
	// - At the leaf case, re-bind user names (excluding "_") to those temps, then run body.
	//
	// This avoids leaking/shadowing user names across branches and allows mixing `a` and `_`
	// in the same variant across different tuple cases.
	var buildSwitch func(level int, subset []tupleCase, env [][]string) *SwitchStmt
	buildSwitch = func(level int, subset []tupleCase, env [][]string) *SwitchStmt {
		sw := new(SwitchStmt)
		sw.pos = pos
		sw.Init = nil
		sw.Tag = NewName(pos, temps[level].Value)

		// Partition by variant key + arity (ignore user binding names).
		type group struct {
			rep   Expr
			key   string
			arity int
			cases []tupleCase
		}
		var order []string
		groups := make(map[string]*group)
		for _, tc := range subset {
			p := tc.pats[level]
			key := variantKey(p)
			arity := variantArity(p)
			if key == "" || arity < 0 {
				return nil
			}
			gk := key + "#" + strconv.Itoa(arity)
			g := groups[gk]
			if g == nil {
				order = append(order, gk)
				g = &group{rep: p, key: key, arity: arity}
				groups[gk] = g
			} else if g.arity != arity || g.key != key {
				return nil
			}
			g.cases = append(g.cases, tc)
		}

		appendEnv := func(env [][]string, payloadTemps []string) [][]string {
			n := make([][]string, 0, len(env)+1)
			n = append(n, env...)
			n = append(n, payloadTemps)
			return n
		}

		genUserBindings := func(tc tupleCase, env [][]string) []Stmt {
			declared := make(map[string]bool)
			var binds []Stmt
			for i, pat := range tc.pats {
				call, ok := pat.(*CallExpr)
				if !ok || call == nil {
					continue // unit variant has no payload
				}
				temps := env[i]
				if len(temps) != len(call.ArgList) {
					return nil
				}
				for j, arg := range call.ArgList {
					n, ok := arg.(*Name)
					if !ok || n == nil {
						return nil
					}
					if n.Value == "_" {
						continue
					}
					as := new(AssignStmt)
					as.pos = tc.pos
					if declared[n.Value] {
						as.Op = 0 // '='
					} else {
						as.Op = Def // ':='
						declared[n.Value] = true
					}
					as.Lhs = NewName(tc.pos, n.Value)
					as.Rhs = NewName(tc.pos, temps[j])
					binds = append(binds, as)
				}
			}
			return binds
		}

		for _, gk := range order {
			g := groups[gk]
			cc := new(CaseClause)
			cc.pos = g.rep.Pos()

			// Build payload temp names for this case clause, and rebuild the case pattern.
			var payloadTemps []string
			if g.arity > 0 {
				payloadTemps = make([]string, g.arity)
				for i := 0; i < g.arity; i++ {
					payloadTemps[i] = r.gensym(pos, "_tuple_payload_"+strconv.Itoa(level)+"_").Value
				}
			}
			cc.Cases = buildVariantPatternWithTemps(g.rep, payloadTemps)
			if cc.Cases == nil {
				return nil
			}

			nextEnv := appendEnv(env, payloadTemps)

			// Leaf level: emit final cases that run the original body.
			if level == len(temps)-1 {
				// At the leaf, each group should correspond to exactly one tupleCase.
				// If not, we'd need additional disambiguation (literals/guards), which we don't support here.
				if len(g.cases) != 1 {
					return nil
				}
				tc := g.cases[0]
				var leafBody []Stmt
				if binds := genUserBindings(tc, nextEnv); binds == nil {
					return nil
				} else {
					leafBody = append(leafBody, binds...)
				}
				leafBody = append(leafBody, tc.body...)
				cc.Body = leafBody
				sw.Body = append(sw.Body, cc)
				continue
			}

			inner := buildSwitch(level+1, g.cases, nextEnv)
			if inner == nil {
				return nil
			}
			inner.Rbrace = s.Rbrace
			cc.Body = []Stmt{inner}
			sw.Body = append(sw.Body, cc)
		}

		// Add a default clause to:
		// - suppress enum exhaustiveness checking in types2
		// - and (if tuple-switch has a user default) route mismatches to that default.
		def := new(CaseClause)
		def.pos = pos
		def.Cases = nil
		if defaultLabel != nil {
			def.Body = []Stmt{gotoLabel(pos, defaultLabel)}
		} else {
			def.Body = nil
		}
		sw.Body = append(sw.Body, def)

		sw.Rbrace = s.Rbrace
		return sw
	}

	outer := buildSwitch(0, cases, nil)
	if outer == nil {
		return nil
	}

	var after []Stmt
	after = append(after, initStmts...)
	after = append(after, outer)

	// If the tuple-switch has a user-provided default, arrange control flow as:
	//   switch ... { ... default: goto _tuple_default }
	//   goto _tuple_end
	// _tuple_default:
	//   <defaultBody>
	// _tuple_end:
	//   (empty)
	if defaultLabel != nil && endLabel != nil {
		after = append(after, gotoLabel(pos, endLabel))

		defBlk := new(BlockStmt)
		defBlk.pos = pos
		defBlk.List = defaultBody
		defBlk.Rbrace = s.Rbrace

		defLab := new(LabeledStmt)
		defLab.pos = pos
		defLab.Label = NewName(pos, defaultLabel.Value)
		defLab.Stmt = defBlk
		after = append(after, defLab)

		endStmt := new(EmptyStmt)
		endStmt.pos = pos

		endLab := new(LabeledStmt)
		endLab.pos = pos
		endLab.Label = NewName(pos, endLabel.Value)
		endLab.Stmt = endStmt
		after = append(after, endLab)
	}

	blk := new(BlockStmt)
	blk.pos = pos

	// Preserve switch init scoping by wrapping into an explicit block.
	if s.Init != nil {
		after = append([]Stmt{r.rewriteSimpleStmtToStmt(s.Init)}, after...)
	}

	blk.List = after
	blk.Rbrace = s.Rbrace
	return blk
}

func (r *tupleSwitchPreRewriter) rewriteSimpleStmtToStmt(s SimpleStmt) Stmt {
	// SimpleStmt is also a Stmt; we just run our simple-stmt rewrite and return it.
	ss := r.rewriteSimpleStmt(s)
	if st, ok := ss.(Stmt); ok {
		return st
	}
	return s
}

func unpackTuple(e Expr) ([]Expr, bool) {
	pe, ok := e.(*ParenExpr)
	if !ok || pe == nil {
		return nil, false
	}
	le, ok := pe.X.(*ListExpr)
	if !ok || le == nil {
		return nil, false
	}
	return le.ElemList, true
}

func variantKey(p Expr) string {
	switch p := p.(type) {
	case *CallExpr:
		return String(p.Fun)
	case *SelectorExpr:
		return String(p)
	default:
		return ""
	}
}

func variantArity(p Expr) int {
	switch p := p.(type) {
	case *CallExpr:
		return len(p.ArgList)
	case *SelectorExpr:
		return 0
	default:
		return -1
	}
}

func isVariantPatternNameArgs(p Expr) bool {
	switch p := p.(type) {
	case *SelectorExpr:
		return true // unit variant
	case *CallExpr:
		if p == nil {
			return false
		}
		for _, a := range p.ArgList {
			if _, ok := a.(*Name); !ok {
				return false
			}
		}
		return true
	default:
		return false
	}
}

// buildVariantPatternWithTemps rebuilds an enum-variant pattern expression using compiler-generated
// payload temp names (so the pattern match binds temps, not user variables).
func buildVariantPatternWithTemps(rep Expr, payloadTemps []string) Expr {
	switch p := rep.(type) {
	case *SelectorExpr:
		// unit variant
		if len(payloadTemps) != 0 {
			return nil
		}
		// Shallow clone to avoid reusing the same node across multiple parents.
		cp := new(SelectorExpr)
		cp.pos = p.Pos()
		cp.X = p.X
		cp.Sel = p.Sel
		return cp
	case *CallExpr:
		if p == nil {
			return nil
		}
		if len(payloadTemps) != len(p.ArgList) {
			return nil
		}
		ce := new(CallExpr)
		ce.pos = p.Pos()
		ce.Fun = p.Fun
		if len(payloadTemps) > 0 {
			ce.ArgList = make([]Expr, len(payloadTemps))
			for i, tn := range payloadTemps {
				ce.ArgList[i] = NewName(p.Pos(), tn)
			}
		}
		return ce
	default:
		return nil
	}
}
