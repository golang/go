// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import "strconv"

// RewriteQuestionExprs rewrites OptionalChainExpr and TernaryExpr
// into standard Go syntax before type checking.
func RewriteQuestionExprs(file *File) {
	r := &rewriter{file: file}
	r.rewriteFile(file)
}

type rewriter struct {
	tempCounter  int
	file         *File
	needsReflect bool
}

func (r *rewriter) rewriteFile(file *File) {
	// First pass: rewrite declarations
	for i, decl := range file.DeclList {
		file.DeclList[i] = r.rewriteDecl(decl)
	}

	// Add reflect import if needed (must be done after rewriting to know if needed)
	if r.needsReflect {
		r.addReflectImport()
	}
}

// addReflectImport adds "reflect" to the file's import declarations
func (r *rewriter) addReflectImport() {
	if r.file == nil {
		return
	}

	// Check if reflect is already imported
	for _, decl := range r.file.DeclList {
		if imp, ok := decl.(*ImportDecl); ok {
			if imp.Path != nil {
				path := imp.Path.Value
				// Remove quotes for comparison
				if path == `"reflect"` || path == "`reflect`" {
					return // Already imported
				}
			}
		}
	}

	// Find position from first existing import or use file position
	var pos Pos
	for _, decl := range r.file.DeclList {
		if imp, ok := decl.(*ImportDecl); ok {
			pos = imp.Pos()
			break
		}
	}

	// Create new import declaration with proper BasicLit
	// The value should be the string literal including quotes
	pathLit := &BasicLit{
		Value: `"reflect"`, // Must include quotes for strconv.Unquote
		Kind:  StringLit,
		Bad:   false,
	}
	pathLit.SetPos(pos)

	importDecl := &ImportDecl{
		Path:         pathLit,
		LocalPkgName: nil, // Use package's default name (reflect)
		Group:        nil, // Not part of a group
	}
	importDecl.SetPos(pos)

	// Find insertion point - after last import
	lastImportIdx := -1
	for i, decl := range r.file.DeclList {
		if _, ok := decl.(*ImportDecl); ok {
			lastImportIdx = i
		}
	}

	if lastImportIdx >= 0 {
		// Insert after last import
		newDecls := make([]Decl, 0, len(r.file.DeclList)+1)
		for i, decl := range r.file.DeclList {
			newDecls = append(newDecls, decl)
			if i == lastImportIdx {
				newDecls = append(newDecls, importDecl)
			}
		}
		r.file.DeclList = newDecls
	} else {
		// No imports - insert at beginning
		newDecls := make([]Decl, 0, len(r.file.DeclList)+1)
		newDecls = append(newDecls, importDecl)
		newDecls = append(newDecls, r.file.DeclList...)
		r.file.DeclList = newDecls
	}
}

func (r *rewriter) rewriteDecl(decl Decl) Decl {
	switch d := decl.(type) {
	case *FuncDecl:
		if d.Body != nil {
			d.Body = r.rewriteBlockStmt(d.Body)
		}
	case *VarDecl:
		if d.Values != nil {
			d.Values = r.rewriteExpr(d.Values)
		}
	case *ConstDecl:
		if d.Values != nil {
			d.Values = r.rewriteExpr(d.Values)
		}
	}
	return decl
}

func (r *rewriter) rewriteBlockStmt(block *BlockStmt) *BlockStmt {
	if block == nil {
		return nil
	}
	for i, stmt := range block.List {
		block.List[i] = r.rewriteStmt(stmt)
	}
	return block
}

func (r *rewriter) rewriteStmt(stmt Stmt) Stmt {
	if stmt == nil {
		return nil
	}
	switch s := stmt.(type) {
	case *ExprStmt:
		s.X = r.rewriteExpr(s.X)
	case *AssignStmt:
		s.Lhs = r.rewriteExpr(s.Lhs)
		s.Rhs = r.rewriteExpr(s.Rhs)
	case *ReturnStmt:
		if s.Results != nil {
			s.Results = r.rewriteExpr(s.Results)
		}
	case *IfStmt:
		if s.Init != nil {
			s.Init = r.rewriteSimpleStmt(s.Init)
		}
		s.Cond = r.rewriteExpr(s.Cond)
		s.Then = r.rewriteBlockStmt(s.Then)
		if s.Else != nil {
			s.Else = r.rewriteStmt(s.Else)
		}
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
		s.Body = r.rewriteBlockStmt(s.Body)
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
	case *SelectStmt:
		for _, cc := range s.Body {
			r.rewriteCommClause(cc)
		}
	case *BlockStmt:
		r.rewriteBlockStmt(s)
	case *DeclStmt:
		for _, d := range s.DeclList {
			r.rewriteDecl(d)
		}
	case *SendStmt:
		s.Chan = r.rewriteExpr(s.Chan)
		s.Value = r.rewriteExpr(s.Value)
	}
	return stmt
}

func (r *rewriter) rewriteSimpleStmt(stmt SimpleStmt) SimpleStmt {
	if stmt == nil {
		return nil
	}
	switch s := stmt.(type) {
	case *ExprStmt:
		s.X = r.rewriteExpr(s.X)
	case *AssignStmt:
		s.Lhs = r.rewriteExpr(s.Lhs)
		s.Rhs = r.rewriteExpr(s.Rhs)
	case *SendStmt:
		s.Chan = r.rewriteExpr(s.Chan)
		s.Value = r.rewriteExpr(s.Value)
	}
	return stmt
}

func (r *rewriter) rewriteCaseClause(cc *CaseClause) {
	if cc.Cases != nil {
		cc.Cases = r.rewriteExpr(cc.Cases)
	}
	for i, stmt := range cc.Body {
		cc.Body[i] = r.rewriteStmt(stmt)
	}
}

func (r *rewriter) rewriteCommClause(cc *CommClause) {
	if cc.Comm != nil {
		cc.Comm = r.rewriteSimpleStmt(cc.Comm)
	}
	for i, stmt := range cc.Body {
		cc.Body[i] = r.rewriteStmt(stmt)
	}
}

func (r *rewriter) rewriteExpr(expr Expr) Expr {
	if expr == nil {
		return nil
	}

	switch e := expr.(type) {
	case *OptionalChainExpr:
		// OptionalChainExpr is now handled in types2/noder stage with type information
		// We only need to recursively rewrite the base expression
		e.X = r.rewriteExpr(e.X)
		return e

	case *TernaryExpr:
		// TernaryExpr is now handled in types2/noder stage with type information
		// This allows proper type inference when both branches have the same type
		e.Cond = r.rewriteExpr(e.Cond)
		e.X = r.rewriteExpr(e.X)
		e.Y = r.rewriteExpr(e.Y)
		return e

	case *Name:
		return e

	case *BasicLit:
		return e

	case *CompositeLit:
		if e.Type != nil {
			e.Type = r.rewriteExpr(e.Type)
		}
		for i, elem := range e.ElemList {
			e.ElemList[i] = r.rewriteExpr(elem)
		}
		return e

	case *KeyValueExpr:
		e.Key = r.rewriteExpr(e.Key)
		e.Value = r.rewriteExpr(e.Value)
		return e

	case *FuncLit:
		if e.Body != nil {
			e.Body = r.rewriteBlockStmt(e.Body)
		}
		return e

	case *ParenExpr:
		e.X = r.rewriteExpr(e.X)
		return e

	case *SelectorExpr:
		e.X = r.rewriteExpr(e.X)
		return e

	case *IndexExpr:
		e.X = r.rewriteExpr(e.X)
		e.Index = r.rewriteExpr(e.Index)
		return e

	case *SliceExpr:
		e.X = r.rewriteExpr(e.X)
		for i, idx := range e.Index {
			if idx != nil {
				e.Index[i] = r.rewriteExpr(idx)
			}
		}
		return e

	case *AssertExpr:
		e.X = r.rewriteExpr(e.X)
		if e.Type != nil {
			e.Type = r.rewriteExpr(e.Type)
		}
		return e

	case *Operation:
		e.X = r.rewriteExpr(e.X)
		if e.Y != nil {
			e.Y = r.rewriteExpr(e.Y)
		}
		return e

	case *CallExpr:
		// Optional chain method calls (x?.method(args)) are handled in types2/noder
		// Just recursively process the expressions here
		e.Fun = r.rewriteExpr(e.Fun)
		for i, arg := range e.ArgList {
			e.ArgList[i] = r.rewriteExpr(arg)
		}
		return e

	case *ListExpr:
		for i, elem := range e.ElemList {
			e.ElemList[i] = r.rewriteExpr(elem)
		}
		return e

	default:
		return expr
	}
}

// rewriteOptionalChainCall converts x?.method(args) to a nil-safe call using reflect:
//
//	func(_v interface{}) interface{} {
//	    if _v == nil { return nil }
//	    rv := reflect.ValueOf(_v)
//	    if rv.Kind() == reflect.Ptr && rv.IsNil() { return nil }
//	    m := rv.MethodByName("method")
//	    results := m.Call([]reflect.Value{...args...})
//	    if len(results) > 0 { return results[0].Interface() }
//	    return nil
//	}(x)
func (r *rewriter) rewriteOptionalChainCall(opt *OptionalChainExpr, call *CallExpr) Expr {
	pos := opt.Pos()
	base := r.rewriteExpr(opt.X)
	methodName := opt.Sel.Value

	// Rewrite arguments
	for i, arg := range call.ArgList {
		call.ArgList[i] = r.rewriteExpr(arg)
	}

	// Mark that we need reflect import
	r.needsReflect = true

	// Build the nil-safe IIFE
	return r.buildReflectMethodCall(pos, base, methodName, call.ArgList)
}

// rewriteOptionalChain converts x?.field to a nil-safe field access using reflect
func (r *rewriter) rewriteOptionalChain(e *OptionalChainExpr) Expr {
	pos := e.Pos()
	base := r.rewriteExpr(e.X)
	fieldName := e.Sel.Value

	// Mark that we need reflect import
	r.needsReflect = true

	// Build the nil-safe field access
	return r.buildReflectFieldAccess(pos, base, fieldName)
}

// buildReflectMethodCall creates a nil-safe method call using reflect
func (r *rewriter) buildReflectMethodCall(pos Pos, base Expr, methodName string, args []Expr) Expr {
	// Create parameter _v
	paramV := r.newName(pos, "_v")
	paramVRef1 := r.newName(pos, "_v")
	paramVRef2 := r.newName(pos, "_v")

	// Create nil checks and reflect call
	stmts := []Stmt{}

	// if _v == nil { return nil }
	stmts = append(stmts, r.createNilCheck(pos, paramVRef1))

	// rv := reflect.ValueOf(_v)
	rvName := r.newName(pos, "_rv")
	rvRef1 := r.newName(pos, "_rv")
	rvRef2 := r.newName(pos, "_rv")
	rvRef3 := r.newName(pos, "_rv")

	reflectValueOf := r.createSelectorExpr(pos, r.newName(pos, "reflect"), "ValueOf")
	valueOfCall := r.createCallExpr(pos, reflectValueOf, []Expr{paramVRef2})
	rvAssign := r.createShortVarDecl(pos, rvName, valueOfCall)
	stmts = append(stmts, rvAssign)

	// if rv.Kind() == reflect.Ptr && rv.IsNil() { return nil }
	kindCall := r.createCallExpr(pos, r.createSelectorExpr(pos, rvRef1, "Kind"), nil)
	reflectPtr := r.createSelectorExpr(pos, r.newName(pos, "reflect"), "Ptr")
	kindEq := r.createBinaryOp(pos, Eql, kindCall, reflectPtr)
	isNilCall := r.createCallExpr(pos, r.createSelectorExpr(pos, rvRef2, "IsNil"), nil)
	andCond := r.createBinaryOp(pos, AndAnd, kindEq, isNilCall)

	nilRet := r.createReturnNil(pos)
	ifPtrNil := r.createIfStmt(pos, andCond, nilRet)
	stmts = append(stmts, ifPtrNil)

	// m := rv.MethodByName("methodName")
	mName := r.newName(pos, "_m")
	mRef := r.newName(pos, "_m")
	methodNameLit := new(BasicLit)
	methodNameLit.SetPos(pos)
	methodNameLit.Kind = StringLit
	methodNameLit.Value = `"` + methodName + `"`
	methodByName := r.createCallExpr(pos, r.createSelectorExpr(pos, rvRef3, "MethodByName"), []Expr{methodNameLit})
	mAssign := r.createShortVarDecl(pos, mName, methodByName)
	stmts = append(stmts, mAssign)

	// results := m.Call([]reflect.Value{args...})
	// For simplicity, we call with empty args if no args
	resultsName := r.newName(pos, "_results")
	resultsRef := r.newName(pos, "_results")

	// Create []reflect.Value{} or with args
	reflectValue := r.createSelectorExpr(pos, r.newName(pos, "reflect"), "Value")
	var argsSlice Expr
	if len(args) == 0 {
		// []reflect.Value{}
		argsSlice = r.createCompositeLit(pos, r.createSliceType(pos, reflectValue), nil)
	} else {
		// []reflect.Value{reflect.ValueOf(arg1), ...}
		argExprs := make([]Expr, len(args))
		for i, arg := range args {
			valueOf := r.createSelectorExpr(pos, r.newName(pos, "reflect"), "ValueOf")
			argExprs[i] = r.createCallExpr(pos, valueOf, []Expr{arg})
		}
		argsSlice = r.createCompositeLit(pos, r.createSliceType(pos, reflectValue), argExprs)
	}

	callMethod := r.createCallExpr(pos, r.createSelectorExpr(pos, mRef, "Call"), []Expr{argsSlice})
	resultsAssign := r.createShortVarDecl(pos, resultsName, callMethod)
	stmts = append(stmts, resultsAssign)

	// if len(results) > 0 { return results[0].Interface() }
	lenCall := r.createCallExpr(pos, r.newName(pos, "len"), []Expr{resultsRef})
	zero := new(BasicLit)
	zero.SetPos(pos)
	zero.Kind = IntLit
	zero.Value = "0"
	lenGt0 := r.createBinaryOp(pos, Gtr, lenCall, zero)

	// Block for if len(results) > 0
	thenStmts := []Stmt{}

	// r0 := results[0]
	r0Name := r.newName(pos, "_r0")
	r0Ref1 := r.newName(pos, "_r0")
	r0Ref2 := r.newName(pos, "_r0")
	r0Ref3 := r.newName(pos, "_r0")
	resultsRef2 := r.newName(pos, "_results")
	zero2 := new(BasicLit)
	zero2.SetPos(pos)
	zero2.Kind = IntLit
	zero2.Value = "0"
	indexExpr := r.createIndexExpr(pos, resultsRef2, zero2)
	r0Assign := r.createShortVarDecl(pos, r0Name, indexExpr)
	thenStmts = append(thenStmts, r0Assign)

	// if r0.Kind() == reflect.Ptr && r0.IsNil() { return nil }
	r0KindCall := r.createCallExpr(pos, r.createSelectorExpr(pos, r0Ref1, "Kind"), nil)
	reflectPtr4 := r.createSelectorExpr(pos, r.newName(pos, "reflect"), "Ptr")
	r0KindEq := r.createBinaryOp(pos, Eql, r0KindCall, reflectPtr4)
	r0IsNilCall := r.createCallExpr(pos, r.createSelectorExpr(pos, r0Ref2, "IsNil"), nil)
	r0AndCond := r.createBinaryOp(pos, AndAnd, r0KindEq, r0IsNilCall)
	nilRet3 := r.createReturnNil(pos)
	ifR0Nil := r.createIfStmt(pos, r0AndCond, nilRet3)
	thenStmts = append(thenStmts, ifR0Nil)

	// return r0.Interface()
	interfaceCall := r.createCallExpr(pos, r.createSelectorExpr(pos, r0Ref3, "Interface"), nil)
	returnResult := r.createReturnStmt(pos, interfaceCall)
	thenStmts = append(thenStmts, returnResult)

	// Create the then block
	thenBlock := new(BlockStmt)
	thenBlock.SetPos(pos)
	thenBlock.List = thenStmts

	ifLenGt0 := new(IfStmt)
	ifLenGt0.SetPos(pos)
	ifLenGt0.Cond = lenGt0
	ifLenGt0.Then = thenBlock

	stmts = append(stmts, ifLenGt0)

	// return nil
	stmts = append(stmts, r.createReturnNil(pos))

	// Create the function
	return r.createIIFE(pos, paramV, stmts, base)
}

// buildReflectFieldAccess creates a nil-safe field access using reflect
func (r *rewriter) buildReflectFieldAccess(pos Pos, base Expr, fieldName string) Expr {
	// Create parameter _v
	paramV := r.newName(pos, "_v")
	paramVRef1 := r.newName(pos, "_v")
	paramVRef2 := r.newName(pos, "_v")

	stmts := []Stmt{}

	// if _v == nil { return nil }
	stmts = append(stmts, r.createNilCheck(pos, paramVRef1))

	// rv := reflect.ValueOf(_v)
	rvName := r.newName(pos, "_rv")
	rvRef1 := r.newName(pos, "_rv")
	rvRef2 := r.newName(pos, "_rv")
	rvRef3 := r.newName(pos, "_rv")

	reflectValueOf := r.createSelectorExpr(pos, r.newName(pos, "reflect"), "ValueOf")
	valueOfCall := r.createCallExpr(pos, reflectValueOf, []Expr{paramVRef2})
	rvAssign := r.createShortVarDecl(pos, rvName, valueOfCall)
	stmts = append(stmts, rvAssign)

	// if rv.Kind() == reflect.Ptr && rv.IsNil() { return nil }
	kindCall := r.createCallExpr(pos, r.createSelectorExpr(pos, rvRef1, "Kind"), nil)
	reflectPtr := r.createSelectorExpr(pos, r.newName(pos, "reflect"), "Ptr")
	kindEq := r.createBinaryOp(pos, Eql, kindCall, reflectPtr)
	isNilCall := r.createCallExpr(pos, r.createSelectorExpr(pos, rvRef2, "IsNil"), nil)
	andCond := r.createBinaryOp(pos, AndAnd, kindEq, isNilCall)

	nilRet := r.createReturnNil(pos)
	ifPtrNil := r.createIfStmt(pos, andCond, nilRet)
	stmts = append(stmts, ifPtrNil)

	// if rv.Kind() == reflect.Ptr { rv = rv.Elem() }
	rvRef4 := r.newName(pos, "_rv")
	rvRef5 := r.newName(pos, "_rv")
	kindCall2 := r.createCallExpr(pos, r.createSelectorExpr(pos, rvRef3, "Kind"), nil)
	reflectPtr2 := r.createSelectorExpr(pos, r.newName(pos, "reflect"), "Ptr")
	kindEqPtr := r.createBinaryOp(pos, Eql, kindCall2, reflectPtr2)
	elemCall := r.createCallExpr(pos, r.createSelectorExpr(pos, rvRef4, "Elem"), nil)
	rvReassign := r.createAssignStmt(pos, rvRef5, elemCall)
	ifPtr := r.createIfStmt(pos, kindEqPtr, rvReassign)
	stmts = append(stmts, ifPtr)

	// f := rv.FieldByName("fieldName")
	fName := r.newName(pos, "_f")
	fRef := r.newName(pos, "_f")
	fRef2 := r.newName(pos, "_f")
	fRef3 := r.newName(pos, "_f")
	rvRef6 := r.newName(pos, "_rv")
	fieldNameLit := new(BasicLit)
	fieldNameLit.SetPos(pos)
	fieldNameLit.Kind = StringLit
	fieldNameLit.Value = `"` + fieldName + `"`
	fieldByName := r.createCallExpr(pos, r.createSelectorExpr(pos, rvRef6, "FieldByName"), []Expr{fieldNameLit})
	fAssign := r.createShortVarDecl(pos, fName, fieldByName)
	stmts = append(stmts, fAssign)

	// if f.Kind() == reflect.Ptr && f.IsNil() { return nil }
	fKindCall := r.createCallExpr(pos, r.createSelectorExpr(pos, fRef, "Kind"), nil)
	reflectPtr3 := r.createSelectorExpr(pos, r.newName(pos, "reflect"), "Ptr")
	fKindEq := r.createBinaryOp(pos, Eql, fKindCall, reflectPtr3)
	fIsNilCall := r.createCallExpr(pos, r.createSelectorExpr(pos, fRef2, "IsNil"), nil)
	fAndCond := r.createBinaryOp(pos, AndAnd, fKindEq, fIsNilCall)

	nilRet2 := r.createReturnNil(pos)
	ifFieldNil := r.createIfStmt(pos, fAndCond, nilRet2)
	stmts = append(stmts, ifFieldNil)

	// return f.Interface()
	interfaceCall := r.createCallExpr(pos, r.createSelectorExpr(pos, fRef3, "Interface"), nil)
	stmts = append(stmts, r.createReturnStmt(pos, interfaceCall))

	return r.createIIFE(pos, paramV, stmts, base)
}

// Helper functions for AST construction
func (r *rewriter) newName(pos Pos, name string) *Name {
	n := new(Name)
	n.SetPos(pos)
	n.Value = name
	return n
}

func (r *rewriter) createSelectorExpr(pos Pos, x Expr, sel string) *SelectorExpr {
	s := new(SelectorExpr)
	s.SetPos(pos)
	s.X = x
	s.Sel = r.newName(pos, sel)
	return s
}

func (r *rewriter) createCallExpr(pos Pos, fun Expr, args []Expr) *CallExpr {
	c := new(CallExpr)
	c.SetPos(pos)
	c.Fun = fun
	c.ArgList = args
	return c
}

func (r *rewriter) createBinaryOp(pos Pos, op Operator, x, y Expr) *Operation {
	o := new(Operation)
	o.SetPos(pos)
	o.Op = op
	o.X = x
	o.Y = y
	return o
}

func (r *rewriter) createShortVarDecl(pos Pos, name *Name, value Expr) *AssignStmt {
	a := new(AssignStmt)
	a.SetPos(pos)
	a.Op = Def
	a.Lhs = name
	a.Rhs = value
	return a
}

func (r *rewriter) createAssignStmt(pos Pos, lhs, rhs Expr) *AssignStmt {
	a := new(AssignStmt)
	a.SetPos(pos)
	a.Op = 0 // regular assignment
	a.Lhs = lhs
	a.Rhs = rhs
	return a
}

func (r *rewriter) createIfStmt(pos Pos, cond Expr, body Stmt) *IfStmt {
	i := new(IfStmt)
	i.SetPos(pos)
	i.Cond = cond
	if block, ok := body.(*BlockStmt); ok {
		i.Then = block
	} else {
		block := new(BlockStmt)
		block.SetPos(pos)
		block.List = []Stmt{body}
		i.Then = block
	}
	return i
}

func (r *rewriter) createReturnNil(pos Pos) *ReturnStmt {
	ret := new(ReturnStmt)
	ret.SetPos(pos)
	ret.Results = r.newName(pos, "nil")
	return ret
}

func (r *rewriter) createReturnStmt(pos Pos, result Expr) *ReturnStmt {
	ret := new(ReturnStmt)
	ret.SetPos(pos)
	ret.Results = result
	return ret
}

func (r *rewriter) createNilCheck(pos Pos, v *Name) *IfStmt {
	nilName := r.newName(pos, "nil")
	cond := r.createBinaryOp(pos, Eql, v, nilName)
	return r.createIfStmt(pos, cond, r.createReturnNil(pos))
}

func (r *rewriter) createSliceType(pos Pos, elem Expr) *SliceType {
	s := new(SliceType)
	s.SetPos(pos)
	s.Elem = elem
	return s
}

func (r *rewriter) createCompositeLit(pos Pos, typ Expr, elems []Expr) *CompositeLit {
	c := new(CompositeLit)
	c.SetPos(pos)
	c.Type = typ
	c.ElemList = elems
	return c
}

func (r *rewriter) createIndexExpr(pos Pos, x Expr, index Expr) *IndexExpr {
	i := new(IndexExpr)
	i.SetPos(pos)
	i.X = x
	i.Index = index
	return i
}

func (r *rewriter) createIIFE(pos Pos, param *Name, body []Stmt, arg Expr) *CallExpr {
	// Create interface{} type
	interfaceType := new(InterfaceType)
	interfaceType.SetPos(pos)

	// Create parameter field
	paramField := new(Field)
	paramField.SetPos(pos)
	paramField.Name = param
	paramField.Type = interfaceType

	// Create result field
	resultType := new(InterfaceType)
	resultType.SetPos(pos)
	resultField := new(Field)
	resultField.SetPos(pos)
	resultField.Type = resultType

	// Create function type
	funcType := new(FuncType)
	funcType.SetPos(pos)
	funcType.ParamList = []*Field{paramField}
	funcType.ResultList = []*Field{resultField}

	// Create function body
	funcBody := new(BlockStmt)
	funcBody.SetPos(pos)
	funcBody.List = body

	// Create function literal
	funcLit := new(FuncLit)
	funcLit.SetPos(pos)
	funcLit.Type = funcType
	funcLit.Body = funcBody

	// Create call expression
	call := new(CallExpr)
	call.SetPos(pos)
	call.Fun = funcLit
	call.ArgList = []Expr{arg}

	return call
}

// RewriteDefaultParams rewrites functions with default parameter values
// into standard Go syntax before type checking.
//
// Example transformation:
//
//	func sum(a int = 1, b int = 2) int { return a + b }
//
// becomes:
//
//	func sum(_args ...int) int {
//	    a := 1
//	    if len(_args) > 0 { a = _args[0] }
//	    b := 2
//	    if len(_args) > 1 { b = _args[1] }
//	    return a + b
//	}
//
// For mixed parameters:
//
//	func sum2(a int, b int = 2) int { return a + b }
//
// becomes:
//
//	func sum2(a int, _args ...int) int {
//	    b := 2
//	    if len(_args) > 0 { b = _args[0] }
//	    return a + b
//	}
func RewriteDefaultParams(file *File) {
	d := &defaultParamRewriter{}
	d.rewriteFile(file)
}

type defaultParamRewriter struct{}

func (d *defaultParamRewriter) rewriteFile(file *File) {
	for _, decl := range file.DeclList {
		if fn, ok := decl.(*FuncDecl); ok {
			d.rewriteFuncDecl(fn)
		}
	}
}

func (d *defaultParamRewriter) rewriteFuncDecl(fn *FuncDecl) {
	if fn.Type == nil || fn.Body == nil {
		return
	}

	params := fn.Type.ParamList
	if len(params) == 0 {
		return
	}

	// Find the first parameter with a default value
	firstDefaultIdx := -1
	for i, p := range params {
		if p.DefaultValue != nil {
			firstDefaultIdx = i
			break
		}
	}

	if firstDefaultIdx == -1 {
		return // No default parameters
	}

	// Save original parameter list before rewriting (for decorator use)
	if fn.OrigParamList == nil {
		fn.OrigParamList = make([]*Field, len(params))
		for i, p := range params {
			// Create a copy of the parameter without DefaultValue
			origParam := &Field{
				Name: p.Name,
				Type: p.Type,
			}
			origParam.SetPos(p.Pos())
			fn.OrigParamList[i] = origParam
		}
	}

	// Check that all default parameters have the same type
	// For simplicity, we require all default parameters to have the same type
	defaultParams := params[firstDefaultIdx:]
	var commonType Expr
	for _, p := range defaultParams {
		if commonType == nil {
			commonType = p.Type
		}
		// Note: We're comparing type pointers, which works because
		// consecutive parameters often share the same Type pointer.
		// For a more robust solution, we'd need type comparison.
	}

	pos := fn.Pos()

	// Build new parameter list:
	// - Keep regular parameters (before firstDefaultIdx)
	// - Add a single variadic parameter for all default params
	newParams := make([]*Field, 0, firstDefaultIdx+1)

	// Copy regular parameters
	for i := 0; i < firstDefaultIdx; i++ {
		newParams = append(newParams, params[i])
	}

	// Create variadic parameter: _args ...T
	argsName := NewName(pos, "_args")
	dotsType := new(DotsType)
	dotsType.SetPos(pos)
	dotsType.Elem = commonType

	argsField := new(Field)
	argsField.SetPos(pos)
	argsField.Name = argsName
	argsField.Type = dotsType
	newParams = append(newParams, argsField)

	// Build statements to insert at the beginning of function body
	preamble := make([]Stmt, 0, len(defaultParams)*2)

	for i, p := range defaultParams {
		if p.Name == nil {
			continue
		}

		paramName := p.Name.Value

		// Create: paramName := defaultValue
		initStmt := d.createShortVarDecl(pos, paramName, p.DefaultValue)
		preamble = append(preamble, initStmt)

		// Create: if len(_args) > i { paramName = _args[i] }
		ifStmt := d.createDefaultOverride(pos, paramName, i)
		preamble = append(preamble, ifStmt)
	}

	// Update function type with new parameters
	fn.Type.ParamList = newParams

	// Prepend preamble to function body
	newBody := make([]Stmt, 0, len(preamble)+len(fn.Body.List))
	newBody = append(newBody, preamble...)
	newBody = append(newBody, fn.Body.List...)
	fn.Body.List = newBody
}

func (d *defaultParamRewriter) createShortVarDecl(pos Pos, name string, value Expr) *AssignStmt {
	a := new(AssignStmt)
	a.SetPos(pos)
	a.Op = Def
	a.Lhs = NewName(pos, name)
	a.Rhs = value
	return a
}

func (d *defaultParamRewriter) createDefaultOverride(pos Pos, paramName string, index int) *IfStmt {
	// Build: if len(_args) > index { paramName = _args[index] }

	// len(_args)
	lenCall := new(CallExpr)
	lenCall.SetPos(pos)
	lenCall.Fun = NewName(pos, "len")
	lenCall.ArgList = []Expr{NewName(pos, "_args")}

	// index literal
	indexLit := new(BasicLit)
	indexLit.SetPos(pos)
	indexLit.Kind = IntLit
	indexLit.Value = strconv.Itoa(index)

	// len(_args) > index
	cond := new(Operation)
	cond.SetPos(pos)
	cond.Op = Gtr
	cond.X = lenCall
	cond.Y = indexLit

	// _args[index]
	indexExpr := new(IndexExpr)
	indexExpr.SetPos(pos)
	indexExpr.X = NewName(pos, "_args")
	indexIdx := new(BasicLit)
	indexIdx.SetPos(pos)
	indexIdx.Kind = IntLit
	indexIdx.Value = strconv.Itoa(index)
	indexExpr.Index = indexIdx

	// paramName = _args[index]
	assign := new(AssignStmt)
	assign.SetPos(pos)
	assign.Op = 0 // regular assignment
	assign.Lhs = NewName(pos, paramName)
	assign.Rhs = indexExpr

	// { paramName = _args[index] }
	thenBlock := new(BlockStmt)
	thenBlock.SetPos(pos)
	thenBlock.List = []Stmt{assign}

	// if len(_args) > index { ... }
	ifStmt := new(IfStmt)
	ifStmt.SetPos(pos)
	ifStmt.Cond = cond
	ifStmt.Then = thenBlock

	return ifStmt
}

// ============================================================================
// Method Overloading Support
// ============================================================================

// isOverloadSingleUnderscoreMagic reports whether an overloaded method should
// keep its single underscore prefix when being renamed.
//
// Without this, methods that already start with '_' would be renamed as
// "__name_suffix", which is not desired for magic methods.
func isOverloadSingleUnderscoreMagic(methodName string) bool {
	switch methodName {
	case "_init", "_getitem", "_setitem",
		"_add", "_sub", "_mul", "_div", "_mod",
		"_radd", "_rsub", "_rmul", "_rdiv", "_rmod",
		"_inc", "_dec":
		return true
	default:
		return false
	}
}

// OverloadInfo stores information about overloaded methods for a receiver type
type OverloadInfo struct {
	ReceiverType string           // Receiver type name (e.g., "User" or "*User")
	MethodName   string           // Original method name (e.g., "SayHello")
	Overloads    []*OverloadEntry // All overload versions
}

// OverloadEntry represents a single overload version of a method
type OverloadEntry struct {
	ParamTypes []string  // Parameter type signatures (e.g., ["int"] or ["float64"])
	MinArgs    int       // Minimum number of arguments (considering default params)
	MaxArgs    int       // Maximum number of arguments
	NewName    string    // Renamed method name (e.g., "_SayHello_int")
	OrigDecl   *FuncDecl // Original declaration
}

// PreprocessOverloadedMethods collects overloaded methods, renames them,
// and rewrites call sites based on argument literal types.
// Returns a map of "ReceiverType.MethodName" -> OverloadInfo.
func PreprocessOverloadedMethods(file *File) map[string]*OverloadInfo {
	// Step 1: Collect all methods grouped by (receiver type, method name)
	methodGroups := make(map[string][]*FuncDecl)

	for _, decl := range file.DeclList {
		fn, ok := decl.(*FuncDecl)
		if !ok || fn.Recv == nil {
			continue // Not a method
		}

		recvType := typeExprToString(fn.Recv.Type)
		methodName := fn.Name.Value
		key := recvType + "." + methodName

		methodGroups[key] = append(methodGroups[key], fn)
	}

	// Step 2: Process groups with multiple methods (overloads)
	overloadInfos := make(map[string]*OverloadInfo)

	for key, methods := range methodGroups {
		if len(methods) <= 1 {
			continue // No overloading
		}

		// Extract receiver type and method name from key
		parts := splitFirst(key, ".")
		recvType := parts[0]
		methodName := parts[1]

		info := &OverloadInfo{
			ReceiverType: recvType,
			MethodName:   methodName,
			Overloads:    make([]*OverloadEntry, 0, len(methods)),
		}

		for _, fn := range methods {
			// Generate parameter type signature
			paramTypes := getParamTypeStrings(fn.Type.ParamList)
			suffix := generateMethodSuffix(paramTypes)

			// Special handling for _init and magic methods: keep single underscore prefix
			var newName string
			if isOverloadSingleUnderscoreMagic(methodName) {
				newName = methodName + suffix // Keep single underscore prefix
			} else {
				newName = "_" + methodName + suffix
			}

			// Calculate min/max args considering default parameters
			minArgs := 0
			maxArgs := len(fn.Type.ParamList)
			for i, param := range fn.Type.ParamList {
				if param.DefaultValue == nil {
					minArgs = i + 1
				}
			}

			entry := &OverloadEntry{
				ParamTypes: paramTypes,
				MinArgs:    minArgs,
				MaxArgs:    maxArgs,
				NewName:    newName,
				OrigDecl:   fn,
			}
			info.Overloads = append(info.Overloads, entry)

			// Rename the method definition
			fn.Name.Value = newName
		}

		overloadInfos[key] = info
	}

	// Step 3: Rewrite call sites (before type checking)
	if len(overloadInfos) > 0 {
		rewriter := &overloadPreRewriter{
			overloads:     overloadInfos,
			methodNameMap: buildMethodNameMap(overloadInfos),
		}
		rewriter.rewriteFile(file)
	}

	return overloadInfos
}

// AddReturnToInitMethods adds return values to all _init methods
// This should be called after method overloading but before constructors
func AddReturnToInitMethods(file *File) {
	for _, decl := range file.DeclList {
		fn, ok := decl.(*FuncDecl)
		if !ok || fn.Recv == nil {
			continue
		}

		methodName := fn.Name.Value
		// Check if this is an _init method (original or renamed by overloading)
		if methodName == "_init" || (len(methodName) >= 6 && methodName[:6] == "_init_") {
			addReturnToInitMethod(fn)
		}
	}
}

// addReturnToInitMethod adds return type and return statement to _init methods
func addReturnToInitMethod(fn *FuncDecl) {
	if fn.Body == nil || fn.Type == nil || fn.Recv == nil {
		return
	}

	// Skip if already has return values
	if fn.Type.ResultList != nil && len(fn.Type.ResultList) > 0 {
		return
	}

	pos := fn.Pos()
	recvType := fn.Recv.Type

	// Only support pointer receivers for _init
	if op, ok := recvType.(*Operation); !ok || op.Op != Mul || op.Y != nil {
		// Not a pointer receiver - skip
		return
	}

	// Create result field with pointer type
	resultField := new(Field)
	resultField.SetPos(pos)
	resultField.Type = recvType // Already a pointer type

	// Add return type to function
	fn.Type.ResultList = []*Field{resultField}

	// Add "return receiver" at the end of body
	recvName := fn.Recv.Name.Value
	returnStmt := new(ReturnStmt)
	returnStmt.SetPos(pos)
	returnStmt.Results = NewName(pos, recvName)

	// Append return statement to body
	fn.Body.List = append(fn.Body.List, returnStmt)
}

// buildMethodNameMap creates a map from original method name to overload info
func buildMethodNameMap(overloads map[string]*OverloadInfo) map[string][]*OverloadInfo {
	result := make(map[string][]*OverloadInfo)
	for _, info := range overloads {
		result[info.MethodName] = append(result[info.MethodName], info)
	}
	return result
}

// overloadPreRewriter rewrites method calls before type checking
type overloadPreRewriter struct {
	overloads     map[string]*OverloadInfo
	methodNameMap map[string][]*OverloadInfo
}

func (r *overloadPreRewriter) rewriteFile(file *File) {
	for _, decl := range file.DeclList {
		r.rewriteDeclPre(decl)
	}
}

func (r *overloadPreRewriter) rewriteDeclPre(decl Decl) {
	switch d := decl.(type) {
	case *FuncDecl:
		if d.Body != nil {
			r.rewriteBlockStmtPre(d.Body)
		}
	case *VarDecl:
		if d.Values != nil {
			r.rewriteExprPre(d.Values)
		}
	}
}

func (r *overloadPreRewriter) rewriteBlockStmtPre(block *BlockStmt) {
	if block == nil {
		return
	}
	for _, stmt := range block.List {
		r.rewriteStmtPre(stmt)
	}
}

func (r *overloadPreRewriter) rewriteStmtPre(stmt Stmt) {
	if stmt == nil {
		return
	}
	switch s := stmt.(type) {
	case *ExprStmt:
		r.rewriteExprPre(s.X)
	case *AssignStmt:
		r.rewriteExprPre(s.Lhs)
		r.rewriteExprPre(s.Rhs)
	case *ReturnStmt:
		if s.Results != nil {
			r.rewriteExprPre(s.Results)
		}
	case *IfStmt:
		if s.Init != nil {
			r.rewriteSimpleStmtPre(s.Init)
		}
		r.rewriteExprPre(s.Cond)
		r.rewriteBlockStmtPre(s.Then)
		if s.Else != nil {
			r.rewriteStmtPre(s.Else)
		}
	case *ForStmt:
		if s.Init != nil {
			r.rewriteSimpleStmtPre(s.Init)
		}
		if s.Cond != nil {
			r.rewriteExprPre(s.Cond)
		}
		if s.Post != nil {
			r.rewriteSimpleStmtPre(s.Post)
		}
		r.rewriteBlockStmtPre(s.Body)
	case *SwitchStmt:
		if s.Init != nil {
			r.rewriteSimpleStmtPre(s.Init)
		}
		if s.Tag != nil {
			r.rewriteExprPre(s.Tag)
		}
		for _, cc := range s.Body {
			r.rewriteCaseClausePre(cc)
		}
	case *BlockStmt:
		r.rewriteBlockStmtPre(s)
	case *DeclStmt:
		for _, d := range s.DeclList {
			r.rewriteDeclPre(d)
		}
	}
}

func (r *overloadPreRewriter) rewriteSimpleStmtPre(stmt SimpleStmt) {
	if stmt == nil {
		return
	}
	switch s := stmt.(type) {
	case *ExprStmt:
		r.rewriteExprPre(s.X)
	case *AssignStmt:
		r.rewriteExprPre(s.Lhs)
		r.rewriteExprPre(s.Rhs)
	}
}

func (r *overloadPreRewriter) rewriteCaseClausePre(cc *CaseClause) {
	if cc.Cases != nil {
		r.rewriteExprPre(cc.Cases)
	}
	for _, stmt := range cc.Body {
		r.rewriteStmtPre(stmt)
	}
}

func (r *overloadPreRewriter) rewriteExprPre(expr Expr) {
	if expr == nil {
		return
	}

	switch e := expr.(type) {
	case *CallExpr:
		// Check if this is a method call on an overloaded method
		if sel, ok := e.Fun.(*SelectorExpr); ok {
			r.tryRewriteMethodCallPre(e, sel)
		}
		// Recursively process
		r.rewriteExprPre(e.Fun)
		for _, arg := range e.ArgList {
			r.rewriteExprPre(arg)
		}

	case *FuncLit:
		if e.Body != nil {
			r.rewriteBlockStmtPre(e.Body)
		}

	case *ParenExpr:
		r.rewriteExprPre(e.X)

	case *SelectorExpr:
		r.rewriteExprPre(e.X)

	case *IndexExpr:
		r.rewriteExprPre(e.X)
		r.rewriteExprPre(e.Index)

	case *Operation:
		r.rewriteExprPre(e.X)
		if e.Y != nil {
			r.rewriteExprPre(e.Y)
		}

	case *ListExpr:
		for _, elem := range e.ElemList {
			r.rewriteExprPre(elem)
		}
	}
}

func (r *overloadPreRewriter) tryRewriteMethodCallPre(call *CallExpr, sel *SelectorExpr) {
	methodName := sel.Sel.Value

	// Check if this method name has any overloads
	infos, ok := r.methodNameMap[methodName]
	if !ok {
		return
	}

	// Infer argument types from literals
	argTypes := make([]string, len(call.ArgList))
	for i, arg := range call.ArgList {
		argTypes[i] = inferLiteralType(arg)
	}

	// Find matching overload
	for _, info := range infos {
		for _, entry := range info.Overloads {
			// Check if argument count is within the valid range
			numArgs := len(argTypes)
			if numArgs < entry.MinArgs || numArgs > entry.MaxArgs {
				continue // Argument count doesn't match
			}

			// Check if argument types match (only check provided arguments)
			match := true
			for i := 0; i < numArgs; i++ {
				if !typeMatchesPre(entry.ParamTypes[i], argTypes[i]) {
					match = false
					break
				}
			}

			if match {
				// Replace the method name
				sel.Sel.Value = entry.NewName
				return
			}
		}
	}
}

// inferLiteralType tries to infer the type of a literal expression
func inferLiteralType(expr Expr) string {
	switch e := expr.(type) {
	case *BasicLit:
		switch e.Kind {
		case IntLit:
			// Check if it contains a decimal point
			if containsDecimalPoint(e.Value) {
				return "float64"
			}
			return "int"
		case FloatLit:
			return "float64"
		case ImagLit:
			return "complex128"
		case RuneLit:
			return "rune"
		case StringLit:
			return "string"
		}
	case *CompositeLit:
		// Handle composite literals with explicit types, e.g. []int{1, 2}
		// This is important for magic method / overload resolution which runs
		// before full type checking.
		if e.Type != nil {
			return typeExprToString(e.Type)
		}
	case *Name:
		// Could be a variable - we don't know the type yet
		// Return "unknown" and try to match based on other criteria
		if e.Value == "true" || e.Value == "false" {
			return "bool"
		}
		if e.Value == "nil" {
			return "nil"
		}
		return "unknown:" + e.Value
	case *Operation:
		// Unary operation - infer from operand
		if e.Y == nil {
			return inferLiteralType(e.X)
		}
		// Binary operation - try to infer from operands
		leftType := inferLiteralType(e.X)
		rightType := inferLiteralType(e.Y)
		if leftType == "float64" || rightType == "float64" {
			return "float64"
		}
		if leftType == "int" || rightType == "int" {
			return "int"
		}
		return "unknown"
	case *ParenExpr:
		return inferLiteralType(e.X)
	}
	return "unknown"
}

// canWrapArgsAsIntSlice reports whether it's reasonable to try the "no comma → []int fallback"
// by wrapping the given args into a single []int{...} slice literal.
//
// We keep this conservative: don't attempt to wrap obvious non-integer literals (like "str").
// For names/expressions with unknown types, we allow wrapping and rely on the type checker later.
func canWrapArgsAsIntSlice(args []Expr) bool {
	if len(args) == 0 {
		return false
	}
	for _, a := range args {
		if a == nil {
			return false
		}
		if lit, ok := a.(*BasicLit); ok {
			// Only allow integer literals here. Other literal kinds should not be wrapped into []int.
			if lit.Kind != IntLit {
				return false
			}
		}
	}
	return true
}

// containsDecimalPoint checks if a number literal contains a decimal point
func containsDecimalPoint(s string) bool {
	for _, c := range s {
		if c == '.' {
			return true
		}
	}
	return false
}

// matchParamTypesPreCheck matches parameter types during pre-processing
func matchParamTypesPreCheck(paramTypes, argTypes []string) bool {
	if len(paramTypes) != len(argTypes) {
		return false
	}
	for i := range paramTypes {
		if !typeMatchesPre(paramTypes[i], argTypes[i]) {
			return false
		}
	}
	return true
}

// typeMatchesPre checks if types match during pre-processing
func typeMatchesPre(paramType, argType string) bool {
	// Exact match
	if paramType == argType {
		return true
	}

	// interface{} and any match any type
	if paramType == "interface{}" || paramType == "any" {
		return true
	}

	// int literal can match various int types
	if argType == "int" {
		switch paramType {
		case "int", "int8", "int16", "int32", "int64",
			"uint", "uint8", "uint16", "uint32", "uint64",
			"interface{}", "any":
			return true
		}
	}

	// float64 literal matches float types
	if argType == "float64" {
		switch paramType {
		case "float32", "float64", "interface{}", "any":
			return true
		}
	}

	// string matches interface{} and any
	if argType == "string" && (paramType == "interface{}" || paramType == "any") {
		return true
	}

	// Unknown type - we'll be optimistic and check param count only
	if len(argType) > 8 && argType[:8] == "unknown:" {
		return true
	}
	if argType == "unknown" {
		return true
	}

	return false
}

// typeExprToString converts a type expression to a string representation
func typeExprToString(expr Expr) string {
	if expr == nil {
		return ""
	}

	switch t := expr.(type) {
	case *Name:
		return t.Value
	case *Operation:
		if t.Op == Mul && t.Y == nil {
			// Pointer type: *T
			return "*" + typeExprToString(t.X)
		}
	case *SelectorExpr:
		// Qualified type: pkg.Type
		return typeExprToString(t.X) + "." + t.Sel.Value
	case *IndexExpr:
		// Generic type: Type[T]
		return typeExprToString(t.X) + "[" + typeExprToString(t.Index) + "]"
	case *SliceType:
		return "[]" + typeExprToString(t.Elem)
	case *ArrayType:
		if t.Len != nil {
			return "[...]" + typeExprToString(t.Elem)
		}
		return "[]" + typeExprToString(t.Elem)
	case *DotsType:
		// Variadic parameter: ...T
		return "..." + typeExprToString(t.Elem)
	case *MapType:
		return "map[" + typeExprToString(t.Key) + "]" + typeExprToString(t.Value)
	case *ChanType:
		prefix := "chan "
		if t.Dir == SendOnly {
			prefix = "chan<- "
		} else if t.Dir == RecvOnly {
			prefix = "<-chan "
		}
		return prefix + typeExprToString(t.Elem)
	case *InterfaceType:
		return "interface{}"
	case *FuncType:
		return "func(...)"
	}

	return "unknown"
}

// getParamTypeStrings extracts parameter type strings from a parameter list
func getParamTypeStrings(params []*Field) []string {
	var types []string
	for _, p := range params {
		typeStr := typeExprToString(p.Type)
		// If multiple names share the same type, add one entry per name
		if p.Name != nil {
			types = append(types, typeStr)
		} else {
			// Anonymous parameter
			types = append(types, typeStr)
		}
	}
	return types
}

// generateMethodSuffix creates a unique suffix based on parameter types
func generateMethodSuffix(paramTypes []string) string {
	if len(paramTypes) == 0 {
		return "_void"
	}
	result := ""
	for _, pt := range paramTypes {
		// Sanitize type string for use in identifier
		sanitized := sanitizeTypeName(pt)
		result += "_" + sanitized
	}
	return result
}

// sanitizeTypeName converts a type string to a valid identifier suffix
func sanitizeTypeName(typeName string) string {
	// Handle variadic parameters first: ...T -> variadic_T
	if len(typeName) >= 3 && typeName[:3] == "..." {
		return "variadic_" + sanitizeTypeName(typeName[3:])
	}

	// Replace special characters
	result := ""
	for _, ch := range typeName {
		switch ch {
		case '*':
			result += "ptr"
		case '[':
			result += "slice"
		case ']':
			// skip
		case '.':
			result += "_"
		case ' ':
			// skip
		default:
			result += string(ch)
		}
	}
	return result
}

// splitFirst splits a string by the first occurrence of sep
func splitFirst(s, sep string) []string {
	for i := 0; i < len(s); i++ {
		if i+len(sep) <= len(s) && s[i:i+len(sep)] == sep {
			return []string{s[:i], s[i+len(sep):]}
		}
	}
	return []string{s, ""}
}

// ============================================================================
// Arithmetic Operator Overloading Support
// ============================================================================

// RewriteArithmeticOperators rewrites arithmetic operators into magic method calls.
//
// Supported operators:
//   - Binary: + - * / %  ->  a._add(b) / a._sub(b) / ...
//   - Reverse fallback:  if a doesn't have forward op, try b._radd(a) / ...
//   - Inc/Dec: x++ / x-- -> x._inc() / x._dec()  (only forward; no reverse)
//
// This pass runs before type checking. It is conservative: it only rewrites
// when it can infer that the involved receiver type defines the corresponding
// magic method(s) in the current file.
func RewriteArithmeticOperators(file *File) {
	if file == nil {
		return
	}
	r := &arithOpRewriter{
		arithMethods:    make(map[string]map[string][]*FuncDecl),
		varTypes:        make(map[string]string),
		funcReturnTypes: make(map[string]string),
	}
	r.collectArithmeticMethods(file)
	if len(r.arithMethods) == 0 {
		return
	}
	r.collectFunctionReturnTypes(file)
	r.rewriteFile(file)
}

type arithOpRewriter struct {
	// arithMethods maps: receiverTypeName -> baseMagicMethodName -> candidate method decls.
	// receiverTypeName has no leading '*'.
	arithMethods map[string]map[string][]*FuncDecl
	varTypes     map[string]string // var name -> typeName (no leading '*')

	funcReturnTypes map[string]string // function name -> return typeName (no leading '*')

	insideArithMethod bool
}

func (r *arithOpRewriter) collectArithmeticMethods(file *File) {
	for _, decl := range file.DeclList {
		fn, ok := decl.(*FuncDecl)
		if !ok || fn.Recv == nil || fn.Name == nil {
			continue
		}
		methodName := fn.Name.Value
		if !isArithmeticMagicMethodName(methodName) {
			continue
		}

		recvType := typeExprToString(fn.Recv.Type)
		if len(recvType) > 0 && recvType[0] == '*' {
			recvType = recvType[1:]
		}
		if recvType == "" {
			continue
		}
		m, ok := r.arithMethods[recvType]
		if !ok {
			m = make(map[string][]*FuncDecl)
			r.arithMethods[recvType] = m
		}
		base := arithMagicBaseName(methodName)
		if base == "" {
			continue
		}
		m[base] = append(m[base], fn)
	}
}

func (r *arithOpRewriter) collectFunctionReturnTypes(file *File) {
	for _, decl := range file.DeclList {
		fn, ok := decl.(*FuncDecl)
		if !ok || fn.Recv != nil || fn.Name == nil || fn.Type == nil {
			continue
		}
		if fn.Type.ResultList == nil || len(fn.Type.ResultList) != 1 {
			continue
		}
		res := fn.Type.ResultList[0]
		if res.Type == nil {
			continue
		}
		retType := typeExprToString(res.Type)
		if len(retType) > 0 && retType[0] == '*' {
			retType = retType[1:]
		}
		if retType == "" {
			continue
		}
		if _, exists := r.arithMethods[retType]; exists {
			r.funcReturnTypes[fn.Name.Value] = retType
		}
	}
}

func (r *arithOpRewriter) rewriteFile(file *File) {
	for _, decl := range file.DeclList {
		r.rewriteDecl(decl)
	}
}

func (r *arithOpRewriter) rewriteDecl(decl Decl) {
	switch d := decl.(type) {
	case *FuncDecl:
		if d.Body == nil {
			return
		}

		// Track function parameters
		if d.Type != nil && d.Type.ParamList != nil {
			for _, param := range d.Type.ParamList {
				if param.Name == nil || param.Type == nil {
					continue
				}
				paramType := typeExprToString(param.Type)
				if len(paramType) > 0 && paramType[0] == '*' {
					paramType = paramType[1:]
				}
				if _, exists := r.arithMethods[paramType]; exists {
					r.varTypes[param.Name.Value] = paramType
				}
			}
		}

		// Track receiver variable
		if d.Recv != nil && d.Recv.Name != nil {
			recvType := typeExprToString(d.Recv.Type)
			if len(recvType) > 0 && recvType[0] == '*' {
				recvType = recvType[1:]
			}
			if _, exists := r.arithMethods[recvType]; exists {
				r.varTypes[d.Recv.Name.Value] = recvType
			}
		}

		// Don't rewrite inside arithmetic magic method bodies to avoid recursion.
		wasInside := r.insideArithMethod
		if d.Recv != nil && d.Name != nil && isArithmeticMagicMethodName(d.Name.Value) {
			r.insideArithMethod = true
		}
		r.rewriteBlockStmt(d.Body)
		r.insideArithMethod = wasInside

	case *VarDecl:
		r.trackVarDeclTypes(d)
		if d.Values != nil {
			d.Values = r.rewriteExpr(d.Values)
		}
	case *ConstDecl:
		if d.Values != nil {
			d.Values = r.rewriteExpr(d.Values)
		}
	}
}

func (r *arithOpRewriter) rewriteBlockStmt(block *BlockStmt) {
	if block == nil {
		return
	}
	for i, stmt := range block.List {
		block.List[i] = r.rewriteStmt(stmt)
	}
}

func (r *arithOpRewriter) rewriteStmt(stmt Stmt) Stmt {
	if stmt == nil {
		return nil
	}
	switch s := stmt.(type) {
	case *ExprStmt:
		s.X = r.rewriteExpr(s.X)
		return s
	case *AssignStmt:
		// Track short variable declarations.
		if s.Op == Def {
			r.trackShortVarDeclTypes(s)
		}

		// Inc/Dec: Lhs++ / Lhs--
		if s.Rhs == nil && (s.Op == Add || s.Op == Sub) {
			return r.rewriteIncDecStmt(s)
		}

		s.Lhs = r.rewriteExpr(s.Lhs)
		s.Rhs = r.rewriteExpr(s.Rhs)
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
		s.Cond = r.rewriteExpr(s.Cond)
		r.rewriteBlockStmt(s.Then)
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
		r.rewriteBlockStmt(s.Body)
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
		return s
	case *SelectStmt:
		for _, cc := range s.Body {
			r.rewriteCommClause(cc)
		}
		return s
	case *BlockStmt:
		r.rewriteBlockStmt(s)
		return s
	case *DeclStmt:
		for _, d := range s.DeclList {
			r.rewriteDecl(d)
		}
		return s
	case *SendStmt:
		s.Chan = r.rewriteExpr(s.Chan)
		s.Value = r.rewriteExpr(s.Value)
		return s
	default:
		return stmt
	}
}

func (r *arithOpRewriter) rewriteSimpleStmt(stmt SimpleStmt) SimpleStmt {
	if stmt == nil {
		return nil
	}
	switch s := stmt.(type) {
	case *ExprStmt:
		s.X = r.rewriteExpr(s.X)
		return s
	case *AssignStmt:
		rewritten := r.rewriteStmt(s)
		if simple, ok := rewritten.(SimpleStmt); ok {
			return simple
		}
		// Fallback: keep as-is (should not happen often).
		s.Lhs = r.rewriteExpr(s.Lhs)
		s.Rhs = r.rewriteExpr(s.Rhs)
		return s
	case *SendStmt:
		s.Chan = r.rewriteExpr(s.Chan)
		s.Value = r.rewriteExpr(s.Value)
		return s
	default:
		return stmt
	}
}

func (r *arithOpRewriter) rewriteCaseClause(cc *CaseClause) {
	if cc == nil {
		return
	}
	if cc.Cases != nil {
		cc.Cases = r.rewriteExpr(cc.Cases)
	}
	for i, st := range cc.Body {
		cc.Body[i] = r.rewriteStmt(st)
	}
}

func (r *arithOpRewriter) rewriteCommClause(cc *CommClause) {
	if cc == nil {
		return
	}
	if cc.Comm != nil {
		cc.Comm = r.rewriteSimpleStmt(cc.Comm)
	}
	for i, st := range cc.Body {
		cc.Body[i] = r.rewriteStmt(st)
	}
}

func (r *arithOpRewriter) rewriteExpr(expr Expr) Expr {
	if expr == nil {
		return nil
	}

	switch e := expr.(type) {
	case *Operation:
		// Only handle binary arithmetic operators.
		if e.Y == nil {
			e.X = r.rewriteExpr(e.X)
			return e
		}

		// Rewrite children first.
		left := r.rewriteExpr(e.X)
		right := r.rewriteExpr(e.Y)
		e.X, e.Y = left, right

		if r.insideArithMethod {
			return e
		}

		fwd, rev := arithMethodNamesForOp(e.Op)
		if fwd == "" {
			return e
		}

		leftType := r.tryInferStructTypeName(left)
		rightType := r.tryInferStructTypeName(right)

		// Prefer forward: a._add(b) (only if signature plausibly matches)
		if leftType != "" && r.hasArithBinary(leftType, fwd, rightType) {
			return r.createMethodCall(e.Pos(), left, fwd, []Expr{right})
		}
		// Fallback to reverse: b._radd(a)
		if rev != "" && rightType != "" && r.hasArithBinary(rightType, rev, leftType) {
			return r.createMethodCall(e.Pos(), right, rev, []Expr{left})
		}

		return e

	case *CallExpr:
		e.Fun = r.rewriteExpr(e.Fun)
		for i, a := range e.ArgList {
			e.ArgList[i] = r.rewriteExpr(a)
		}
		return e

	case *SelectorExpr:
		e.X = r.rewriteExpr(e.X)
		return e

	case *ParenExpr:
		e.X = r.rewriteExpr(e.X)
		return e

	case *IndexExpr:
		e.X = r.rewriteExpr(e.X)
		e.Index = r.rewriteExpr(e.Index)
		return e

	case *SliceExpr:
		e.X = r.rewriteExpr(e.X)
		for i, idx := range e.Index {
			if idx != nil {
				e.Index[i] = r.rewriteExpr(idx)
			}
		}
		return e

	case *CompositeLit:
		if e.Type != nil {
			e.Type = r.rewriteExpr(e.Type)
		}
		for i, elem := range e.ElemList {
			e.ElemList[i] = r.rewriteExpr(elem)
		}
		return e

	case *KeyValueExpr:
		e.Key = r.rewriteExpr(e.Key)
		e.Value = r.rewriteExpr(e.Value)
		return e

	case *FuncLit:
		if e.Body != nil {
			r.rewriteBlockStmt(e.Body)
		}
		return e

	case *AssertExpr:
		e.X = r.rewriteExpr(e.X)
		if e.Type != nil {
			e.Type = r.rewriteExpr(e.Type)
		}
		return e

	case *ListExpr:
		for i, elem := range e.ElemList {
			e.ElemList[i] = r.rewriteExpr(elem)
		}
		return e

	default:
		return expr
	}
}

func (r *arithOpRewriter) rewriteIncDecStmt(stmt *AssignStmt) Stmt {
	// Only rewrite if the LHS looks assignable (to preserve Go's checks reasonably).
	if !isAssignableLike(stmt.Lhs) {
		return stmt
	}

	var method string
	switch stmt.Op {
	case Add:
		method = "_inc"
	case Sub:
		method = "_dec"
	default:
		return stmt
	}

	lhs := r.rewriteExpr(stmt.Lhs)
	if r.insideArithMethod {
		stmt.Lhs = lhs
		return stmt
	}
	lhsType := r.tryInferStructTypeName(lhs)
	if lhsType == "" || !r.hasArithNoArg(lhsType, method) {
		stmt.Lhs = lhs
		return stmt
	}

	call := r.createMethodCall(stmt.Pos(), lhs, method, nil)
	es := new(ExprStmt)
	es.SetPos(stmt.Pos())
	es.X = call
	return es
}

func (r *arithOpRewriter) hasArithNoArg(recvType string, methodName string) bool {
	m, ok := r.arithMethods[recvType]
	if !ok {
		return false
	}
	cands := m[methodName]
	return len(cands) > 0
}

// hasArithBinary reports whether recvType has a candidate method for methodName
// whose (single) parameter can plausibly accept argType.
//
// argType is inferred syntactically and has no leading '*'. Empty argType means unknown.
func (r *arithOpRewriter) hasArithBinary(recvType string, methodName string, argType string) bool {
	m, ok := r.arithMethods[recvType]
	if !ok {
		return false
	}
	cands := m[methodName]
	if len(cands) == 0 {
		return false
	}
	// Unknown arg type: be optimistic (we only rewrite when receiver is known anyway).
	if argType == "" {
		return true
	}
	for _, fn := range cands {
		if fn == nil || fn.Type == nil {
			continue
		}
		params := fn.Type.ParamList
		if len(params) != 1 {
			continue
		}
		p := params[0]
		if p == nil || p.Type == nil {
			continue
		}
		pt := typeExprToString(p.Type)
		if len(pt) > 0 && pt[0] == '*' {
			pt = pt[1:]
		}
		// any/interface{} accept any type
		if pt == "any" || pt == "interface{}" {
			return true
		}
		if pt == argType {
			return true
		}
	}
	return false
}

func (r *arithOpRewriter) createMethodCall(pos Pos, base Expr, methodName string, args []Expr) *CallExpr {
	sel := new(SelectorExpr)
	sel.SetPos(pos)
	sel.X = base
	sel.Sel = NewName(pos, methodName)

	call := new(CallExpr)
	call.SetPos(pos)
	call.Fun = sel
	call.ArgList = args
	return call
}

// trackVarDeclTypes records variable types from var declarations.
func (r *arithOpRewriter) trackVarDeclTypes(d *VarDecl) {
	if d == nil {
		return
	}
	if d.Type != nil {
		typeName := typeExprToString(d.Type)
		if len(typeName) > 0 && typeName[0] == '*' {
			typeName = typeName[1:]
		}
		if _, exists := r.arithMethods[typeName]; exists {
			for _, n := range d.NameList {
				r.varTypes[n.Value] = typeName
			}
		}
		return
	}
	if d.Values != nil {
		r.trackAssignmentTypes(d.NameList, d.Values)
	}
}

func (r *arithOpRewriter) trackShortVarDeclTypes(stmt *AssignStmt) {
	if stmt == nil {
		return
	}
	var names []*Name
	switch lhs := stmt.Lhs.(type) {
	case *Name:
		names = []*Name{lhs}
	case *ListExpr:
		for _, elem := range lhs.ElemList {
			if n, ok := elem.(*Name); ok {
				names = append(names, n)
			}
		}
	default:
		return
	}
	r.trackAssignmentTypes(names, stmt.Rhs)
}

func (r *arithOpRewriter) trackAssignmentTypes(names []*Name, values Expr) {
	if len(names) == 0 {
		return
	}
	if len(names) == 1 {
		typeName := r.tryInferStructTypeName(values)
		if typeName != "" {
			if _, exists := r.arithMethods[typeName]; exists {
				r.varTypes[names[0].Value] = typeName
			}
		}
		return
	}
	if list, ok := values.(*ListExpr); ok {
		for i, n := range names {
			if i >= len(list.ElemList) {
				break
			}
			typeName := r.tryInferStructTypeName(list.ElemList[i])
			if typeName != "" {
				if _, exists := r.arithMethods[typeName]; exists {
					r.varTypes[n.Value] = typeName
				}
			}
		}
	}
}

// tryInferStructTypeName attempts to infer the (non-pointer) struct type name
// from an expression, using only local syntactic information.
func (r *arithOpRewriter) tryInferStructTypeName(expr Expr) string {
	switch e := expr.(type) {
	case *Name:
		if t, ok := r.varTypes[e.Value]; ok {
			return t
		}
		return ""
	case *ParenExpr:
		return r.tryInferStructTypeName(e.X)
	case *Operation:
		// &T{} or *p
		if e.Op == And && e.Y == nil {
			return r.tryInferStructTypeName(e.X)
		}
		if e.Op == Mul && e.Y == nil {
			return r.tryInferStructTypeName(e.X)
		}
		return ""
	case *CompositeLit:
		if e.Type != nil {
			t := typeExprToString(e.Type)
			if len(t) > 0 && t[0] == '*' {
				t = t[1:]
			}
			return t
		}
		return ""
	case *CallExpr:
		// Try to infer from method call: x._add(y) etc.
		// This enables chaining like: a + b + a  ->  a._add(b)._add(a)
		if sel, ok := e.Fun.(*SelectorExpr); ok && sel.Sel != nil {
			baseMagic := arithMagicBaseName(sel.Sel.Value)
			if baseMagic != "" {
				recvType := r.tryInferStructTypeName(sel.X)
				if recvType != "" {
					if r.methodReturnsReceiverType(recvType, baseMagic) {
						return recvType
					}
				}
			}
		}
		if name, ok := e.Fun.(*Name); ok {
			if name.Value == "make" && len(e.ArgList) > 0 {
				t := typeExprToString(e.ArgList[0])
				if len(t) > 0 && t[0] == '*' {
					t = t[1:]
				}
				return t
			}
			if rt, ok := r.funcReturnTypes[name.Value]; ok {
				return rt
			}
		}
		return ""
	default:
		return ""
	}
}

// methodReturnsReceiverType reports whether receiver type's magic method returns
// the receiver type (either as value or pointer). This is used only for
// syntactic inference to enable chaining rewrites.
func (r *arithOpRewriter) methodReturnsReceiverType(recvType string, baseMagic string) bool {
	m, ok := r.arithMethods[recvType]
	if !ok {
		return false
	}
	cands := m[baseMagic]
	for _, fn := range cands {
		if fn == nil || fn.Type == nil || fn.Type.ResultList == nil || len(fn.Type.ResultList) != 1 {
			continue
		}
		res := fn.Type.ResultList[0]
		if res == nil || res.Type == nil {
			continue
		}
		rt := typeExprToString(res.Type)
		if rt == recvType {
			return true
		}
		// Allow *T returning to infer T.
		if len(rt) > 0 && rt[0] == '*' && rt[1:] == recvType {
			return true
		}
	}
	return false
}

func arithMethodNamesForOp(op Operator) (forward string, reverse string) {
	switch op {
	case Add:
		return "_add", "_radd"
	case Sub:
		return "_sub", "_rsub"
	case Mul:
		return "_mul", "_rmul"
	case Div:
		return "_div", "_rdiv"
	case Rem:
		return "_mod", "_rmod"
	default:
		return "", ""
	}
}

func isArithmeticMagicMethodName(name string) bool {
	return isArithmeticMagicBase(name) || arithMagicBaseName(name) != ""
}

func isArithmeticMagicBase(name string) bool {
	switch name {
	case "_add", "_sub", "_mul", "_div", "_mod",
		"_radd", "_rsub", "_rmul", "_rdiv", "_rmod",
		"_inc", "_dec":
		return true
	default:
		return false
	}
}

// arithMagicBaseName returns the base magic name if name is a magic method or
// its overloaded form (e.g. "_add_int" -> "_add").
func arithMagicBaseName(name string) string {
	if isArithmeticMagicBase(name) {
		return name
	}
	// Overloaded form: base + "_" + suffix
	bases := []string{
		"_add", "_sub", "_mul", "_div", "_mod",
		"_radd", "_rsub", "_rmul", "_rdiv", "_rmod",
		"_inc", "_dec",
	}
	for _, b := range bases {
		if len(name) > len(b) && name[:len(b)] == b && name[len(b)] == '_' {
			return b
		}
	}
	return ""
}

func isAssignableLike(e Expr) bool {
	switch x := e.(type) {
	case *Name:
		return true
	case *SelectorExpr:
		return true
	case *IndexExpr:
		return true
	case *SliceExpr:
		return true
	case *ParenExpr:
		return isAssignableLike(x.X)
	case *Operation:
		// *p
		return x.Op == Mul && x.Y == nil && isAssignableLike(x.X)
	default:
		return false
	}
}

// RewriteOverloadedCalls replaces method calls with the correct overloaded version
// based on argument types determined during type checking.
// This function should be called after types2 type checking.
func RewriteOverloadedCalls(file *File, overloads map[string]*OverloadInfo) {
	if len(overloads) == 0 {
		return
	}

	rewriter := &overloadCallRewriter{overloads: overloads}
	rewriter.rewriteFile(file)
}

type overloadCallRewriter struct {
	overloads map[string]*OverloadInfo
}

func (r *overloadCallRewriter) rewriteFile(file *File) {
	for _, decl := range file.DeclList {
		r.rewriteDecl(decl)
	}
}

func (r *overloadCallRewriter) rewriteDecl(decl Decl) {
	switch d := decl.(type) {
	case *FuncDecl:
		if d.Body != nil {
			r.rewriteBlockStmt(d.Body)
		}
	case *VarDecl:
		if d.Values != nil {
			r.rewriteExpr(d.Values)
		}
	case *ConstDecl:
		if d.Values != nil {
			r.rewriteExpr(d.Values)
		}
	}
}

func (r *overloadCallRewriter) rewriteBlockStmt(block *BlockStmt) {
	if block == nil {
		return
	}
	for _, stmt := range block.List {
		r.rewriteStmt(stmt)
	}
}

func (r *overloadCallRewriter) rewriteStmt(stmt Stmt) {
	if stmt == nil {
		return
	}
	switch s := stmt.(type) {
	case *ExprStmt:
		r.rewriteExpr(s.X)
	case *AssignStmt:
		r.rewriteExpr(s.Lhs)
		r.rewriteExpr(s.Rhs)
	case *ReturnStmt:
		if s.Results != nil {
			r.rewriteExpr(s.Results)
		}
	case *IfStmt:
		if s.Init != nil {
			r.rewriteSimpleStmt(s.Init)
		}
		r.rewriteExpr(s.Cond)
		r.rewriteBlockStmt(s.Then)
		if s.Else != nil {
			r.rewriteStmt(s.Else)
		}
	case *ForStmt:
		if s.Init != nil {
			r.rewriteSimpleStmt(s.Init)
		}
		if s.Cond != nil {
			r.rewriteExpr(s.Cond)
		}
		if s.Post != nil {
			r.rewriteSimpleStmt(s.Post)
		}
		r.rewriteBlockStmt(s.Body)
	case *SwitchStmt:
		if s.Init != nil {
			r.rewriteSimpleStmt(s.Init)
		}
		if s.Tag != nil {
			r.rewriteExpr(s.Tag)
		}
		for _, cc := range s.Body {
			r.rewriteCaseClause(cc)
		}
	case *SelectStmt:
		for _, cc := range s.Body {
			r.rewriteCommClause(cc)
		}
	case *BlockStmt:
		r.rewriteBlockStmt(s)
	case *DeclStmt:
		for _, d := range s.DeclList {
			r.rewriteDecl(d)
		}
	case *SendStmt:
		r.rewriteExpr(s.Chan)
		r.rewriteExpr(s.Value)
	}
}

func (r *overloadCallRewriter) rewriteSimpleStmt(stmt SimpleStmt) {
	if stmt == nil {
		return
	}
	switch s := stmt.(type) {
	case *ExprStmt:
		r.rewriteExpr(s.X)
	case *AssignStmt:
		r.rewriteExpr(s.Lhs)
		r.rewriteExpr(s.Rhs)
	case *SendStmt:
		r.rewriteExpr(s.Chan)
		r.rewriteExpr(s.Value)
	}
}

func (r *overloadCallRewriter) rewriteCaseClause(cc *CaseClause) {
	if cc.Cases != nil {
		r.rewriteExpr(cc.Cases)
	}
	for _, stmt := range cc.Body {
		r.rewriteStmt(stmt)
	}
}

func (r *overloadCallRewriter) rewriteCommClause(cc *CommClause) {
	if cc.Comm != nil {
		r.rewriteSimpleStmt(cc.Comm)
	}
	for _, stmt := range cc.Body {
		r.rewriteStmt(stmt)
	}
}

func (r *overloadCallRewriter) rewriteExpr(expr Expr) {
	if expr == nil {
		return
	}

	switch e := expr.(type) {
	case *CallExpr:
		// Check if this is a method call on an overloaded method
		if sel, ok := e.Fun.(*SelectorExpr); ok {
			r.tryRewriteMethodCall(e, sel)
		}
		// Recursively process the function and arguments
		r.rewriteExpr(e.Fun)
		for _, arg := range e.ArgList {
			r.rewriteExpr(arg)
		}

	case *CompositeLit:
		if e.Type != nil {
			r.rewriteExpr(e.Type)
		}
		for _, elem := range e.ElemList {
			r.rewriteExpr(elem)
		}

	case *KeyValueExpr:
		r.rewriteExpr(e.Key)
		r.rewriteExpr(e.Value)

	case *FuncLit:
		if e.Body != nil {
			r.rewriteBlockStmt(e.Body)
		}

	case *ParenExpr:
		r.rewriteExpr(e.X)

	case *SelectorExpr:
		r.rewriteExpr(e.X)

	case *IndexExpr:
		r.rewriteExpr(e.X)
		r.rewriteExpr(e.Index)

	case *SliceExpr:
		r.rewriteExpr(e.X)
		for _, idx := range e.Index {
			if idx != nil {
				r.rewriteExpr(idx)
			}
		}

	case *AssertExpr:
		r.rewriteExpr(e.X)
		if e.Type != nil {
			r.rewriteExpr(e.Type)
		}

	case *Operation:
		r.rewriteExpr(e.X)
		if e.Y != nil {
			r.rewriteExpr(e.Y)
		}

	case *ListExpr:
		for _, elem := range e.ElemList {
			r.rewriteExpr(elem)
		}
	}
}

func (r *overloadCallRewriter) tryRewriteMethodCall(call *CallExpr, sel *SelectorExpr) {
	methodName := sel.Sel.Value

	// Get the receiver type from the type info stored in the expression
	recvTypeInfo := sel.X.GetTypeInfo()
	if recvTypeInfo.Type == nil {
		return // No type info available
	}

	recvTypeStr := recvTypeInfo.Type.String()

	// Try to find matching overload info
	// We need to check both the exact type and pointer variations
	key := recvTypeStr + "." + methodName
	overloadInfo := r.overloads[key]

	// If not found, try with pointer prefix removed or added
	if overloadInfo == nil {
		// Try without pointer
		if len(recvTypeStr) > 0 && recvTypeStr[0] == '*' {
			key = recvTypeStr[1:] + "." + methodName
			overloadInfo = r.overloads[key]
		}
		// Try with pointer
		if overloadInfo == nil {
			key = "*" + recvTypeStr + "." + methodName
			overloadInfo = r.overloads[key]
		}
	}

	if overloadInfo == nil {
		return // Not an overloaded method
	}

	// Get argument types from type info
	argTypes := make([]string, len(call.ArgList))
	for i, arg := range call.ArgList {
		argTypeInfo := arg.GetTypeInfo()
		if argTypeInfo.Type != nil {
			argTypes[i] = argTypeInfo.Type.String()
		} else {
			argTypes[i] = "unknown"
		}
	}

	// Find matching overload
	for _, entry := range overloadInfo.Overloads {
		if matchParamTypes(entry.ParamTypes, argTypes) {
			// Replace the method name with the renamed version
			sel.Sel.Value = entry.NewName
			return
		}
	}
}

// matchParamTypes checks if the argument types match the parameter types
func matchParamTypes(paramTypes, argTypes []string) bool {
	if len(paramTypes) != len(argTypes) {
		return false
	}
	for i := range paramTypes {
		if !typesMatch(paramTypes[i], argTypes[i]) {
			return false
		}
	}
	return true
}

// typesMatch checks if two type strings represent compatible types
func typesMatch(paramType, argType string) bool {
	// Exact match
	if paramType == argType {
		return true
	}

	// Handle untyped constants and basic type compatibility
	// e.g., "untyped int" matches "int", "untyped float" matches "float64"
	switch argType {
	case "untyped int":
		return paramType == "int" || paramType == "int8" || paramType == "int16" ||
			paramType == "int32" || paramType == "int64" || paramType == "uint" ||
			paramType == "uint8" || paramType == "uint16" || paramType == "uint32" ||
			paramType == "uint64" || paramType == "float32" || paramType == "float64"
	case "untyped float":
		return paramType == "float32" || paramType == "float64"
	case "untyped string":
		return paramType == "string"
	case "untyped bool":
		return paramType == "bool"
	}

	return false
}

// ============================================================================
// Constructor Support (_init methods)
// ============================================================================

// RewriteConstructors rewrites constructor syntax to standard Go code.
//
// It transforms:
//
//	ds := make(DataStoreInt, "age", 18)
//
// into:
//
//	ds := (&DataStoreInt{})._init("age", 18)  // or _init_xxx if overloaded
//
// Note: This should run AFTER method overloading and default params,
// so _init methods have already been renamed and modified.
func RewriteConstructors(file *File) {
	// Step 1: Find all types with _init methods (including overloaded/renamed ones)
	initMethods := make(map[string][]*FuncDecl) // typename -> list of _init methods

	for _, decl := range file.DeclList {
		fn, ok := decl.(*FuncDecl)
		if !ok || fn.Recv == nil {
			continue
		}

		// Check if method name is "_init" or starts with "_init_" (overloaded)
		methodName := fn.Name.Value
		if methodName != "_init" && (len(methodName) < 6 || methodName[:6] != "_init_") {
			continue
		}

		// Get receiver type name
		recvType := typeExprToString(fn.Recv.Type)
		// Remove * prefix if pointer receiver
		if len(recvType) > 0 && recvType[0] == '*' {
			recvType = recvType[1:]
		}

		initMethods[recvType] = append(initMethods[recvType], fn)
	}

	// Step 2: NO NEED to modify _init methods - they've already been processed by other rewriters

	// Step 3: Rewrite constructor calls
	if len(initMethods) > 0 {
		rewriter := &constructorRewriter{initMethods: initMethods}
		rewriter.rewriteFile(file)
	}
}

// constructorRewriter rewrites constructor calls
type constructorRewriter struct {
	initMethods map[string][]*FuncDecl // typename -> list of _init methods
}

func (r *constructorRewriter) rewriteFile(file *File) {
	for _, decl := range file.DeclList {
		r.rewriteDecl(decl)
	}
}

func (r *constructorRewriter) rewriteDecl(decl Decl) {
	switch d := decl.(type) {
	case *FuncDecl:
		if d.Body != nil {
			r.rewriteBlockStmt(d.Body)
		}
	case *VarDecl:
		if d.Values != nil {
			d.Values = r.rewriteExpr(d.Values)
		}
	}
}

func (r *constructorRewriter) rewriteBlockStmt(block *BlockStmt) {
	if block == nil {
		return
	}
	for i, stmt := range block.List {
		block.List[i] = r.rewriteStmt(stmt)
	}
}

func (r *constructorRewriter) rewriteStmt(stmt Stmt) Stmt {
	if stmt == nil {
		return nil
	}
	switch s := stmt.(type) {
	case *ExprStmt:
		s.X = r.rewriteExpr(s.X)
	case *AssignStmt:
		s.Lhs = r.rewriteExpr(s.Lhs)
		s.Rhs = r.rewriteExpr(s.Rhs)
	case *ReturnStmt:
		if s.Results != nil {
			s.Results = r.rewriteExpr(s.Results)
		}
	case *IfStmt:
		if s.Init != nil {
			s.Init = r.rewriteSimpleStmt(s.Init)
		}
		s.Cond = r.rewriteExpr(s.Cond)
		r.rewriteBlockStmt(s.Then)
		if s.Else != nil {
			s.Else = r.rewriteStmt(s.Else)
		}
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
		r.rewriteBlockStmt(s.Body)
	case *BlockStmt:
		r.rewriteBlockStmt(s)
	case *DeclStmt:
		for _, d := range s.DeclList {
			r.rewriteDecl(d)
		}
	}
	return stmt
}

func (r *constructorRewriter) rewriteSimpleStmt(stmt SimpleStmt) SimpleStmt {
	if stmt == nil {
		return nil
	}
	switch s := stmt.(type) {
	case *ExprStmt:
		s.X = r.rewriteExpr(s.X)
	case *AssignStmt:
		s.Lhs = r.rewriteExpr(s.Lhs)
		s.Rhs = r.rewriteExpr(s.Rhs)
	}
	return stmt
}

func (r *constructorRewriter) rewriteExpr(expr Expr) Expr {
	if expr == nil {
		return nil
	}

	switch e := expr.(type) {
	case *CallExpr:
		// Check if this is: make(TypeName, args...)
		// where TypeName has an _init method
		if makeName, ok := e.Fun.(*Name); ok && makeName.Value == "make" {
			// This is a make call
			if len(e.ArgList) > 0 {
				// First argument should be the type name
				if typeName, ok := e.ArgList[0].(*Name); ok {
					typeNameStr := typeName.Value
					// Check if this type has an _init method
					if _, hasInit := r.initMethods[typeNameStr]; hasInit {
						// Extract constructor arguments (skip the first arg which is the type)
						constructorArgs := e.ArgList[1:]
						// Rewrite: make(TypeName, args...) -> (&TypeName{})._init(args...)
						return r.rewriteConstructorCallWithArgs(e, typeNameStr, constructorArgs)
					}
				}
			}
		}

		// Recursively process function and arguments
		e.Fun = r.rewriteExpr(e.Fun)
		for i, arg := range e.ArgList {
			e.ArgList[i] = r.rewriteExpr(arg)
		}
		return e

	case *ListExpr:
		for i, elem := range e.ElemList {
			e.ElemList[i] = r.rewriteExpr(elem)
		}
		return e

	case *ParenExpr:
		e.X = r.rewriteExpr(e.X)
		return e

	case *Operation:
		e.X = r.rewriteExpr(e.X)
		if e.Y != nil {
			e.Y = r.rewriteExpr(e.Y)
		}
		return e

	case *CompositeLit:
		if e.Type != nil {
			e.Type = r.rewriteExpr(e.Type)
		}
		for i, elem := range e.ElemList {
			e.ElemList[i] = r.rewriteExpr(elem)
		}
		return e

	default:
		return expr
	}
}

func (r *constructorRewriter) rewriteConstructorCall(call *CallExpr, typeName string) Expr {
	return r.rewriteConstructorCallWithArgs(call, typeName, call.ArgList)
}

func (r *constructorRewriter) rewriteConstructorCallWithArgs(call *CallExpr, typeName string, args []Expr) Expr {
	pos := call.Pos()

	// Get init methods for this type
	methods := r.initMethods[typeName]
	if len(methods) == 0 {
		return call // Should not happen
	}

	// Determine which _init method to call
	var methodName string
	if len(methods) == 1 {
		// Only one _init method
		methodName = methods[0].Name.Value
	} else {
		// Multiple _init methods (overloaded) - need to find the right one
		// Use simple type inference based on argument literals
		argTypes := make([]string, len(args))
		for i, arg := range args {
			argTypes[i] = inferLiteralType(arg)
		}

		// Find matching method
		for _, method := range methods {
			paramTypes := getParamTypeStrings(method.Type.ParamList)

			// Check if argument count matches (considering MinArgs/MaxArgs would be ideal)
			if len(argTypes) != len(paramTypes) {
				continue
			}

			// Check if types match
			match := true
			for i := 0; i < len(argTypes); i++ {
				if !typeMatchesPre(paramTypes[i], argTypes[i]) {
					match = false
					break
				}
			}

			if match {
				methodName = method.Name.Value
				break
			}
		}

		// If no match found, use the first one (will fail at type checking)
		if methodName == "" {
			methodName = methods[0].Name.Value
		}
	}

	// Create: &TypeName{}
	typeNameExpr := NewName(pos, typeName)

	// Create composite literal: TypeName{}
	compositeLit := new(CompositeLit)
	compositeLit.SetPos(pos)
	compositeLit.Type = typeNameExpr
	compositeLit.ElemList = nil // empty

	// Create address-of operation: &TypeName{}
	addrOp := new(Operation)
	addrOp.SetPos(pos)
	addrOp.Op = And
	addrOp.X = compositeLit

	// Wrap in parentheses: (&TypeName{})
	parenExpr := new(ParenExpr)
	parenExpr.SetPos(pos)
	parenExpr.X = addrOp

	// Create selector: (&TypeName{})._init or (&TypeName{})._init_xxx
	selector := new(SelectorExpr)
	selector.SetPos(pos)
	selector.X = parenExpr
	selector.Sel = NewName(pos, methodName)

	// Create call: (&TypeName{})._init(args)
	newCall := new(CallExpr)
	newCall.SetPos(pos)
	newCall.Fun = selector
	newCall.ArgList = args

	return newCall
}

// ============================================================================
// Magic Methods Support (_getitem and _setitem)
// ============================================================================

// RewriteMagicMethods rewrites indexing operations (a[x]) and index assignments (a[x] = y)
// to method calls (_getitem and _setitem) when the receiver type has these methods defined.
//
// Example transformations:
//
//	ds[2]        -> ds._getitem(2)
//	ds[1:2]      -> ds._getitem(1, 2)
//	ds[1:2:"str"]-> ds._getitem(1, 2, "str")
//	ds[2] = 5    -> ds._setitem(2, 5)
//	ds[1:2] = 5  -> ds._setitem(1, 2, 5)
func RewriteMagicMethods(file *File) {
	// Step 1: Collect all types with _getitem or _setitem methods
	magicMethods := make(map[string]*MagicMethodInfo)

	for _, decl := range file.DeclList {
		fn, ok := decl.(*FuncDecl)
		if !ok || fn.Recv == nil {
			continue
		}

		methodName := fn.Name.Value
		// Check for _getitem or _setitem methods (including overloaded versions)
		if !isMagicMethodName(methodName) {
			continue
		}

		// Get receiver type name
		recvType := typeExprToString(fn.Recv.Type)
		// Remove * prefix if pointer receiver
		if len(recvType) > 0 && recvType[0] == '*' {
			recvType = recvType[1:]
		}

		info, exists := magicMethods[recvType]
		if !exists {
			info = &MagicMethodInfo{
				TypeName:       recvType,
				GetItemMethods: make([]*FuncDecl, 0),
				SetItemMethods: make([]*FuncDecl, 0),
			}
			magicMethods[recvType] = info
		}

		if isGetItemMethod(methodName) {
			info.GetItemMethods = append(info.GetItemMethods, fn)
		} else if isSetItemMethod(methodName) {
			info.SetItemMethods = append(info.SetItemMethods, fn)
		}
	}

	// Step 2: Rewrite indexing operations in the file, BUT skip magic method bodies
	if len(magicMethods) > 0 {
		rewriter := &magicMethodRewriter{
			magicMethods:      magicMethods,
			insideMagicMethod: false,
			varTypes:          make(map[string]string),
			funcReturnTypes:   make(map[string]string),
		}
		// First pass: collect function return types
		rewriter.collectFunctionReturnTypes(file)
		// Second pass: rewrite magic method calls
		rewriter.rewriteFile(file)
	}
}

// MagicMethodInfo stores information about magic methods for a type
type MagicMethodInfo struct {
	TypeName       string
	GetItemMethods []*FuncDecl
	SetItemMethods []*FuncDecl
}

func isMagicMethodName(name string) bool {
	return isGetItemMethod(name) || isSetItemMethod(name)
}

func isGetItemMethod(name string) bool {
	return name == "_getitem" || (len(name) >= 9 && name[:9] == "_getitem_")
}

func isSetItemMethod(name string) bool {
	return name == "_setitem" || (len(name) >= 9 && name[:9] == "_setitem_")
}

type magicMethodRewriter struct {
	magicMethods      map[string]*MagicMethodInfo
	insideMagicMethod bool              // Track if we're inside a _getitem or _setitem method
	varTypes          map[string]string // Track variable name -> struct type name
	funcReturnTypes   map[string]string // Track function name -> return type name
}

// collectFunctionReturnTypes scans all function declarations and records their return types
func (r *magicMethodRewriter) collectFunctionReturnTypes(file *File) {
	for _, decl := range file.DeclList {
		if fn, ok := decl.(*FuncDecl); ok {
			// Skip methods (they have receivers)
			if fn.Recv != nil {
				continue
			}

			// Check if this function returns a magic method type
			if fn.Type != nil && fn.Type.ResultList != nil && len(fn.Type.ResultList) > 0 {
				// We only care about functions that return a single value
				if len(fn.Type.ResultList) == 1 {
					result := fn.Type.ResultList[0]
					if result.Type != nil {
						returnType := typeExprToString(result.Type)
						// Remove * prefix
						if len(returnType) > 0 && returnType[0] == '*' {
							returnType = returnType[1:]
						}
						// Only record if it's a magic method type
						if _, exists := r.magicMethods[returnType]; exists {
							r.funcReturnTypes[fn.Name.Value] = returnType
						}
					}
				}
			}
		}
	}
}

func (r *magicMethodRewriter) rewriteFile(file *File) {
	for _, decl := range file.DeclList {
		r.rewriteDecl(decl)
	}
}

func (r *magicMethodRewriter) rewriteDecl(decl Decl) {
	switch d := decl.(type) {
	case *FuncDecl:
		if d.Body != nil {
			// Track function parameters
			if d.Type != nil && d.Type.ParamList != nil {
				for _, param := range d.Type.ParamList {
					if param.Name != nil && param.Type != nil {
						paramType := typeExprToString(param.Type)
						// Remove * prefix
						if len(paramType) > 0 && paramType[0] == '*' {
							paramType = paramType[1:]
						}
						if _, exists := r.magicMethods[paramType]; exists {
							r.varTypes[param.Name.Value] = paramType
						}
					}
				}
			}

			// Track receiver type for methods
			if d.Recv != nil && d.Recv.Name != nil {
				recvType := typeExprToString(d.Recv.Type)
				// Remove * prefix
				if len(recvType) > 0 && recvType[0] == '*' {
					recvType = recvType[1:]
				}
				// Add receiver to varTypes so it can be recognized in method body
				if _, exists := r.magicMethods[recvType]; exists {
					r.varTypes[d.Recv.Name.Value] = recvType
				}
			}

			// Check if this is a magic method - if so, don't rewrite its body
			wasMagic := r.insideMagicMethod
			if d.Recv != nil && isMagicMethodName(d.Name.Value) {
				r.insideMagicMethod = true
			}
			r.rewriteBlockStmt(d.Body)
			r.insideMagicMethod = wasMagic
		}
	case *VarDecl:
		// Track variable types from declarations like: var ds *DataStore = ...
		r.trackVarDeclTypes(d)
		if d.Values != nil {
			d.Values = r.rewriteExpr(d.Values)
		}
	}
}

// trackVarDeclTypes records variable types from var declarations
func (r *magicMethodRewriter) trackVarDeclTypes(d *VarDecl) {
	if d.Type != nil {
		// Explicit type: var ds *DataStore
		typeName := typeExprToString(d.Type)
		// Remove * prefix
		if len(typeName) > 0 && typeName[0] == '*' {
			typeName = typeName[1:]
		}
		if _, exists := r.magicMethods[typeName]; exists {
			for _, name := range d.NameList {
				r.varTypes[name.Value] = typeName
			}
		}
	} else if d.Values != nil {
		// Inferred type: var ds = &DataStore{}
		r.trackAssignmentTypes(d.NameList, d.Values)
	}
}

// trackAssignmentTypes records variable types from assignment expressions
func (r *magicMethodRewriter) trackAssignmentTypes(names []*Name, values Expr) {
	if len(names) == 0 {
		return
	}

	// Handle single assignment
	if len(names) == 1 {
		typeName := r.tryInferStructTypeName(values)
		if typeName != "" {
			if _, exists := r.magicMethods[typeName]; exists {
				r.varTypes[names[0].Value] = typeName
			}
		}
		return
	}

	// Handle multiple assignment (e.g., a, b := x, y)
	if listExpr, ok := values.(*ListExpr); ok {
		for i, name := range names {
			if i < len(listExpr.ElemList) {
				typeName := r.tryInferStructTypeName(listExpr.ElemList[i])
				if typeName != "" {
					if _, exists := r.magicMethods[typeName]; exists {
						r.varTypes[name.Value] = typeName
					}
				}
			}
		}
	}
}

// trackShortVarDeclTypes records variable types from short variable declarations
// e.g., ds := &DataStore{...} or m := NewMatrix()
func (r *magicMethodRewriter) trackShortVarDeclTypes(stmt *AssignStmt) {
	// Get LHS names
	var names []*Name
	switch lhs := stmt.Lhs.(type) {
	case *Name:
		names = []*Name{lhs}
	case *ListExpr:
		for _, elem := range lhs.ElemList {
			if name, ok := elem.(*Name); ok {
				names = append(names, name)
			}
		}
	default:
		return
	}

	r.trackAssignmentTypes(names, stmt.Rhs)
}

func (r *magicMethodRewriter) rewriteBlockStmt(block *BlockStmt) {
	if block == nil {
		return
	}
	// Process statements and potentially replace them
	newList := make([]Stmt, 0, len(block.List))
	for _, stmt := range block.List {
		rewritten := r.rewriteStmt(stmt)
		if rewritten != nil {
			newList = append(newList, rewritten)
		}
	}
	block.List = newList
}

func (r *magicMethodRewriter) rewriteStmt(stmt Stmt) Stmt {
	if stmt == nil {
		return nil
	}
	switch s := stmt.(type) {
	case *ExprStmt:
		s.X = r.rewriteExpr(s.X)
		return s
	case *AssignStmt:
		// Check if this is an indexed assignment: a[x] = y
		return r.rewriteAssignStmt(s)
	case *ReturnStmt:
		if s.Results != nil {
			s.Results = r.rewriteExpr(s.Results)
		}
		return s
	case *IfStmt:
		if s.Init != nil {
			s.Init = r.rewriteSimpleStmt(s.Init)
		}
		s.Cond = r.rewriteExpr(s.Cond)
		r.rewriteBlockStmt(s.Then)
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
		r.rewriteBlockStmt(s.Body)
		return s
	case *BlockStmt:
		r.rewriteBlockStmt(s)
		return s
	case *DeclStmt:
		for _, d := range s.DeclList {
			r.rewriteDecl(d)
		}
		return s
	}
	return stmt
}

func (r *magicMethodRewriter) rewriteSimpleStmt(stmt SimpleStmt) SimpleStmt {
	if stmt == nil {
		return nil
	}
	switch s := stmt.(type) {
	case *ExprStmt:
		s.X = r.rewriteExpr(s.X)
		return s
	case *AssignStmt:
		// Note: rewriteAssignStmt might convert AssignStmt to ExprStmt
		// which is not a SimpleStmt. For now, we handle this specially.
		rewritten := r.rewriteAssignStmt(s)
		if simpleRewritten, ok := rewritten.(SimpleStmt); ok {
			return simpleRewritten
		}
		// If it's no longer a SimpleStmt, we have a problem
		// For now, just return the original
		s.Lhs = r.rewriteExpr(s.Lhs)
		s.Rhs = r.rewriteExpr(s.Rhs)
		return s
	}
	return stmt
}

// rewriteAssignStmt handles index assignment: a[x] = y -> a._setitem(x, y)
func (r *magicMethodRewriter) rewriteAssignStmt(stmt *AssignStmt) Stmt {
	// Track variable types for short variable declarations: ds := &DataStore{}
	if stmt.Op == Def {
		r.trackShortVarDeclTypes(stmt)
	}

	// Check if LHS is an IndexExpr or SliceExpr
	var indexExpr *IndexExpr
	var sliceExpr *SliceExpr

	switch lhs := stmt.Lhs.(type) {
	case *IndexExpr:
		indexExpr = lhs
	case *SliceExpr:
		sliceExpr = lhs
	default:
		// Not an index assignment, just rewrite the expressions normally
		stmt.Lhs = r.rewriteExpr(stmt.Lhs)
		stmt.Rhs = r.rewriteExpr(stmt.Rhs)
		return stmt
	}

	var baseExpr Expr
	var args []Expr
	var hasComma bool

	if indexExpr != nil {
		baseExpr = indexExpr.X

		// AGGRESSIVE: Check if we should rewrite this
		if !r.shouldRewriteIndex(baseExpr) {
			// It's a built-in type, don't rewrite
			stmt.Lhs = r.rewriteExpr(stmt.Lhs)
			stmt.Rhs = r.rewriteExpr(stmt.Rhs)
			return stmt
		}

		// Parse the index - could be a single value or colon-separated values
		args, hasComma = r.parseIndexArgs(indexExpr.Index)
	} else if sliceExpr != nil {
		baseExpr = sliceExpr.X

		// AGGRESSIVE: Check if we should rewrite this
		if !r.shouldRewriteSlice(sliceExpr) {
			// It's a built-in type, don't rewrite
			stmt.Lhs = r.rewriteExpr(stmt.Lhs)
			stmt.Rhs = r.rewriteExpr(stmt.Rhs)
			return stmt
		}

		// SliceExpr has Index[0], Index[1], Index[2] (for a[x:y:z])
		args = make([]Expr, 0)
		for _, idx := range sliceExpr.Index {
			if idx != nil {
				args = append(args, idx)
			}
		}
		hasComma = false
	}

	// Transform: a[x, y] = z -> a._setitem(x, y, z)
	pos := stmt.Pos()

	// Add the RHS value as the last argument
	args = append(args, r.rewriteExpr(stmt.Rhs))

	// Create method call: base._setitem(args...)
	methodCall := r.createMagicMethodCallWithComma(pos, baseExpr, "_setitem", args, hasComma)

	// Convert to expression statement (since _setitem typically returns nothing)
	exprStmt := new(ExprStmt)
	exprStmt.SetPos(pos)
	exprStmt.X = methodCall

	return exprStmt
}

func (r *magicMethodRewriter) rewriteExpr(expr Expr) Expr {
	if expr == nil {
		return nil
	}

	switch e := expr.(type) {
	case *IndexExpr:
		// AGGRESSIVE STRATEGY: Try to rewrite ALL IndexExpr to method calls
		// If the type doesn't have _getitem, type checker will catch it later
		// But first check if this looks like a built-in type operation
		if r.shouldRewriteIndex(e.X) {
			return r.rewriteIndexToGetItem(e)
		}
		// Otherwise, just recursively rewrite
		e.X = r.rewriteExpr(e.X)
		e.Index = r.rewriteExpr(e.Index)
		return e

	case *SliceExpr:
		// AGGRESSIVE STRATEGY: Try to rewrite slice expressions with multiple indices
		// This handles a[x:y] syntax for multi-parameter _getitem
		if r.shouldRewriteSlice(e) {
			return r.rewriteSliceToGetItem(e)
		}
		// Otherwise, just recursively rewrite
		e.X = r.rewriteExpr(e.X)
		for i, idx := range e.Index {
			if idx != nil {
				e.Index[i] = r.rewriteExpr(idx)
			}
		}
		return e

	case *CallExpr:
		e.Fun = r.rewriteExpr(e.Fun)
		for i, arg := range e.ArgList {
			e.ArgList[i] = r.rewriteExpr(arg)
		}
		return e

	case *ParenExpr:
		e.X = r.rewriteExpr(e.X)
		return e

	case *SelectorExpr:
		e.X = r.rewriteExpr(e.X)
		return e

	case *Operation:
		e.X = r.rewriteExpr(e.X)
		if e.Y != nil {
			e.Y = r.rewriteExpr(e.Y)
		}
		return e

	case *ListExpr:
		for i, elem := range e.ElemList {
			e.ElemList[i] = r.rewriteExpr(elem)
		}
		return e

	case *CompositeLit:
		if e.Type != nil {
			e.Type = r.rewriteExpr(e.Type)
		}
		for i, elem := range e.ElemList {
			e.ElemList[i] = r.rewriteExpr(elem)
		}
		return e

	default:
		return expr
	}
}

// rewriteIndexToGetItem converts a[x] to a._getitem(x)
func (r *magicMethodRewriter) rewriteIndexToGetItem(indexExpr *IndexExpr) Expr {
	pos := indexExpr.Pos()
	baseExpr := r.rewriteExpr(indexExpr.X)

	// Parse the index expression - could be colon-separated values
	args, hasComma := r.parseIndexArgs(indexExpr.Index)

	return r.createMagicMethodCallWithComma(pos, baseExpr, "_getitem", args, hasComma)
}

// rewriteSliceToGetItem converts a[x:y:z] to a._getitem(x, y, z)
func (r *magicMethodRewriter) rewriteSliceToGetItem(sliceExpr *SliceExpr) Expr {
	pos := sliceExpr.Pos()
	baseExpr := r.rewriteExpr(sliceExpr.X)

	args := make([]Expr, 0)
	for _, idx := range sliceExpr.Index {
		if idx != nil {
			args = append(args, r.rewriteExpr(idx))
		}
	}

	// No comma in pure slice syntax
	return r.createMagicMethodCallWithComma(pos, baseExpr, "_getitem", args, false)
}

// parseIndexArgs parses the index expression which might contain colon-separated values
// Returns: (args []Expr, hasComma bool)
// hasComma indicates if the original syntax used commas
// When hasComma is true, each comma-separated element is wrapped into a slice literal
func (r *magicMethodRewriter) parseIndexArgs(index Expr) ([]Expr, bool) {
	// If the index is a ListExpr, it contains comma-separated values
	// e.g., a[x, y, z] would be ListExpr with [x, y, z]
	// or a[1:2, 3:4] would be ListExpr with [SliceExpr{1,2}, SliceExpr{3,4}]
	if listExpr, ok := index.(*ListExpr); ok {
		result := make([]Expr, 0)

		// Process each comma-separated element
		for _, elem := range listExpr.ElemList {
			var elemArgs []Expr

			// Check if element is a SliceExpr (for a[1:2, 3:4] syntax)
			if sliceExpr, ok := elem.(*SliceExpr); ok {
				// Collect all indices from the slice expression
				for _, idx := range sliceExpr.Index {
					if idx != nil {
						elemArgs = append(elemArgs, r.rewriteExpr(idx))
					}
				}
			} else {
				// Single element
				elemArgs = append(elemArgs, r.rewriteExpr(elem))
			}

			// Wrap elemArgs into a slice literal []T{...}
			// We need to create: []int{1, 2} or []int{1}
			sliceLit := r.createSliceLiteral(elemArgs)
			result = append(result, sliceLit)
		}

		return result, true // Has comma
	}

	// Single value - no comma, no wrapping
	return []Expr{r.rewriteExpr(index)}, false
}

// createSliceLiteral creates a slice literal: []int{elem1, elem2, ...}
// We assume int type for now - this works for most numeric indexing cases
func (r *magicMethodRewriter) createSliceLiteral(elements []Expr) Expr {
	if len(elements) == 0 {
		return nil
	}

	pos := elements[0].Pos()

	// Create slice type: []int
	sliceType := new(SliceType)
	sliceType.SetPos(pos)
	sliceType.Elem = NewName(pos, "int")

	// Create composite literal: []int{elements...}
	compositeLit := new(CompositeLit)
	compositeLit.SetPos(pos)
	compositeLit.Type = sliceType
	compositeLit.ElemList = elements

	return compositeLit
}

// createMagicMethodCall creates a method call like: base.methodName(args...)
// It also handles method overloading by selecting the correct overload based on argument types
func (r *magicMethodRewriter) createMagicMethodCall(pos Pos, base Expr, methodName string, args []Expr) *CallExpr {
	return r.createMagicMethodCallWithComma(pos, base, methodName, args, false)
}

// createMagicMethodCallWithComma creates a method call with comma information
// hasComma: true if original syntax used commas (requires []T parameters)
func (r *magicMethodRewriter) createMagicMethodCallWithComma(pos Pos, base Expr, methodName string, args []Expr, hasComma bool) *CallExpr {
	// Try to resolve the correct overloaded method name
	resolvedName := r.resolveMagicMethodOverloadWithComma(base, methodName, args, hasComma)

	// Fallback behavior (README rule):
	// - No comma: prefer T parameters; if no match, fallback to []T by wrapping indices into []int{...}
	//
	// This enables cases like:
	//   - only _getitem(indices []int) exists, but user writes a[i] or a[i:j]
	//   - only _getitem(indices ...[]int) exists, but user writes a[i] or a[i:j]
	//
	// We only attempt this for []int-based index methods to avoid breaking string-key indexing.
	if !hasComma {
		switch methodName {
		case "_getitem":
			// Only attempt fallback if we didn't already resolve to a specific overload.
			if resolvedName == methodName && canWrapArgsAsIntSlice(args) && r.hasSingleIntSliceIndexMethod(base, methodName) {
				wrapped := r.createSliceLiteral(args)
				if wrapped != nil {
					wrappedArgs := []Expr{wrapped}

					// If there are overloads, resolve using slice preference.
					// If there is only a single method, resolveMagicMethodOverloadWithComma returns
					// the original name, but we still want to wrap the args.
					resolved2 := r.resolveMagicMethodOverloadWithComma(base, methodName, wrappedArgs, true)
					if resolved2 != methodName {
						resolvedName = resolved2
					}
					// Apply wrapped args when slice fallback is applicable.
					args = wrappedArgs
				}
			}

		case "_setitem":
			// For assignments, last arg is the value; only wrap the index arguments.
			// Only attempt fallback if we didn't already resolve to a specific overload.
			if resolvedName == methodName && len(args) >= 2 && canWrapArgsAsIntSlice(args[:len(args)-1]) && r.hasSingleIntSliceIndexMethod(base, methodName) {
				indexArgs := args[:len(args)-1]
				valueArg := args[len(args)-1]
				wrapped := r.createSliceLiteral(indexArgs)
				if wrapped != nil {
					wrappedArgs := []Expr{wrapped, valueArg}
					resolved2 := r.resolveMagicMethodOverloadWithComma(base, methodName, wrappedArgs, true)
					if resolved2 != methodName {
						resolvedName = resolved2
					}
					args = wrappedArgs
				}
			}
		}
	}

	// Create selector: base._getitem or base._setitem (or overloaded version)
	selector := new(SelectorExpr)
	selector.SetPos(pos)
	selector.X = base
	selector.Sel = NewName(pos, resolvedName)

	// Create call expression
	call := new(CallExpr)
	call.SetPos(pos)
	call.Fun = selector
	call.ArgList = args

	return call
}

// hasSingleIntSliceIndexMethod reports whether the receiver type has at least one "single-dimension"
// index magic method that accepts a single []int (or ...[]int for _getitem).
//
// This is used to gate the "no comma → []int fallback" wrapping so we don't accidentally wrap
// string-key indexing (e.g. person["name"]) into []int{...}.
func (r *magicMethodRewriter) hasSingleIntSliceIndexMethod(base Expr, methodName string) bool {
	typeName := r.tryInferStructTypeName(base)
	if typeName == "" {
		return false
	}

	// Resolve magic method info with pointer/value fallback, same as resolveMagicMethodOverloadWithComma.
	info, exists := r.magicMethods[typeName]
	if !exists {
		if len(typeName) > 0 && typeName[0] != '*' {
			info, exists = r.magicMethods["*"+typeName]
		} else if len(typeName) > 0 && typeName[0] == '*' {
			info, exists = r.magicMethods[typeName[1:]]
		}
		if !exists {
			return false
		}
	}

	var methods []*FuncDecl
	if methodName == "_getitem" {
		methods = info.GetItemMethods
	} else if methodName == "_setitem" {
		methods = info.SetItemMethods
	} else {
		return false
	}

	for _, fn := range methods {
		if fn == nil || fn.Type == nil {
			continue
		}
		paramTypes := getParamTypeStrings(fn.Type.ParamList)
		switch methodName {
		case "_getitem":
			// Only consider "single slice argument" forms: []int or ...[]int.
			if len(paramTypes) == 1 && (paramTypes[0] == "[]int" || paramTypes[0] == "...[]int") {
				return true
			}
		case "_setitem":
			// Only consider (indices []int, value T) form for fallback wrapping.
			if len(paramTypes) == 2 && paramTypes[0] == "[]int" {
				return true
			}
		}
	}

	return false
}

// resolveMagicMethodOverload finds the correct overloaded method name based on argument types
func (r *magicMethodRewriter) resolveMagicMethodOverload(base Expr, methodName string, args []Expr) string {
	return r.resolveMagicMethodOverloadWithComma(base, methodName, args, false)
}

// resolveMagicMethodOverloadWithComma finds the correct overloaded method with comma info
// hasComma: if true, ONLY match functions with []T parameters
//
//	if false, PREFER T parameters, fallback to []T parameters
func (r *magicMethodRewriter) resolveMagicMethodOverloadWithComma(base Expr, methodName string, args []Expr, hasComma bool) string {
	// Get the type name of the base expression
	typeName := r.tryInferStructTypeName(base)
	if typeName == "" {
		return methodName // Can't infer type, return original name
	}

	// Get the magic method info for this type
	// Try exact match first, then try with/without * prefix
	info, exists := r.magicMethods[typeName]
	if !exists {
		// Try pointer type if we have value type
		if len(typeName) > 0 && typeName[0] != '*' {
			info, exists = r.magicMethods["*"+typeName]
		} else if len(typeName) > 0 && typeName[0] == '*' {
			// Try value type if we have pointer type
			info, exists = r.magicMethods[typeName[1:]]
		}

		if !exists {
			return methodName
		}
	}

	// Get the list of methods to search
	var methods []*FuncDecl
	if methodName == "_getitem" {
		methods = info.GetItemMethods
	} else if methodName == "_setitem" {
		methods = info.SetItemMethods
	} else {
		return methodName
	}

	// If only one method, no overloading needed - use original name
	if len(methods) <= 1 {
		return methodName
	}

	// Multiple overloads - need to find the matching one
	// Infer argument types
	argTypes := make([]string, len(args))
	for i, arg := range args {
		argTypes[i] = inferLiteralType(arg)
	}

	// Find matching method
	// We'll collect candidates and score them
	type candidate struct {
		fn                  *FuncDecl
		paramTypes          []string
		score               int
		requiresSliceParams bool
	}

	candidates := make([]candidate, 0)

	for _, fn := range methods {
		paramTypes := getParamTypeStrings(fn.Type.ParamList)

		// Check for variadic parameter
		isVariadic := len(paramTypes) > 0 && len(paramTypes[len(paramTypes)-1]) >= 3 &&
			paramTypes[len(paramTypes)-1][:3] == "..."

		// Check if this method has slice parameters
		requiresSliceParams := r.methodHasSliceParams(paramTypes, methodName)

		if isVariadic {
			// Variadic function matching
			if r.matchesVariadicMethod(paramTypes, argTypes) {
				score := 100
				// If hasComma and requires slice params, high priority
				// If !hasComma and doesn't require slice params, high priority
				if hasComma == requiresSliceParams {
					score += 50
				}

				candidates = append(candidates, candidate{
					fn:                  fn,
					paramTypes:          paramTypes,
					score:               score,
					requiresSliceParams: requiresSliceParams,
				})
			}
		} else {
			// Non-variadic: exact parameter count required
			if len(argTypes) != len(paramTypes) {
				continue
			}

			// For _setitem, we need special handling:
			// The last parameter is the value to set, which can be of any type
			// So we're more lenient with the last parameter type matching
			isSetitem := methodName == "_setitem"

			match := true
			matchScore := 0
			for i := 0; i < len(argTypes); i++ {
				// For _setitem's last parameter, be lenient with numeric types
				if isSetitem && i == len(argTypes)-1 {
					// Last parameter is the value - accept any numeric type matching
					if isNumericType(paramTypes[i]) && isNumericType(argTypes[i]) {
						matchScore++
						continue
					}
				}

				if !typeMatchesPre(paramTypes[i], argTypes[i]) {
					match = false
					break
				}
				matchScore++
			}

			if match {
				score := matchScore
				// If hasComma and requires slice params, high priority
				// If !hasComma and doesn't require slice params, high priority
				if hasComma == requiresSliceParams {
					score += 50
				}

				candidates = append(candidates, candidate{
					fn:                  fn,
					paramTypes:          paramTypes,
					score:               score,
					requiresSliceParams: requiresSliceParams,
				})
			}
		}
	}

	// Filter and select best candidate
	if len(candidates) > 0 {
		// If hasComma, ONLY consider slice param candidates
		if hasComma {
			sliceCandidates := make([]candidate, 0)
			for _, c := range candidates {
				if c.requiresSliceParams {
					sliceCandidates = append(sliceCandidates, c)
				}
			}
			candidates = sliceCandidates
		}

		// If no comma, prefer non-slice, but allow slice as fallback
		if !hasComma && len(candidates) > 0 {
			// Check if there are non-slice candidates
			nonSliceCandidates := make([]candidate, 0)
			for _, c := range candidates {
				if !c.requiresSliceParams {
					nonSliceCandidates = append(nonSliceCandidates, c)
				}
			}

			// Prefer non-slice if available
			if len(nonSliceCandidates) > 0 {
				candidates = nonSliceCandidates
			}
		}

		// Select best score
		if len(candidates) > 0 {
			best := candidates[0]
			for _, c := range candidates[1:] {
				if c.score > best.score {
					best = c
				}
			}

			suffix := generateMethodSuffix(best.paramTypes)
			return methodName + suffix
		}
	}

	// No match found, return original name (will cause compile error if truly no match)
	return methodName
}

// methodHasSliceParams checks if a method's parameters include slice types
// For _setitem, ignores the last parameter (the value)
func (r *magicMethodRewriter) methodHasSliceParams(paramTypes []string, methodName string) bool {
	isSetitem := methodName == "_setitem"

	for i, pt := range paramTypes {
		// Skip the last parameter for _setitem (it's the value)
		if isSetitem && i == len(paramTypes)-1 {
			continue
		}

		// Remove variadic prefix
		typeStr := pt
		if len(typeStr) >= 3 && typeStr[:3] == "..." {
			typeStr = typeStr[3:]
		}

		// Check if it's a slice type
		if len(typeStr) >= 2 && typeStr[:2] == "[]" {
			return true
		}
	}

	return false
}

// isNumericType checks if a type string represents a numeric type
func isNumericType(typeName string) bool {
	switch typeName {
	case "int", "int8", "int16", "int32", "int64",
		"uint", "uint8", "uint16", "uint32", "uint64",
		"float32", "float64",
		"complex64", "complex128",
		"interface{}", "any": // interface{} and any can hold any type
		return true
	}
	return false
}

// matchesVariadicMethod checks if argument types match a variadic method signature
func (r *magicMethodRewriter) matchesVariadicMethod(paramTypes []string, argTypes []string) bool {
	numParams := len(paramTypes)
	numArgs := len(argTypes)

	if numParams == 0 {
		return false
	}

	// The last parameter is variadic (e.g., "...int")
	variadicType := paramTypes[numParams-1][3:] // Remove "..." prefix

	// Check non-variadic parameters first
	for i := 0; i < numParams-1; i++ {
		if i >= numArgs {
			return false
		}
		if !typeMatchesPre(paramTypes[i], argTypes[i]) {
			return false
		}
	}

	// All remaining arguments (from numParams-1 onwards) must match the variadic type
	for i := numParams - 1; i < numArgs; i++ {
		if !typeMatchesPre(variadicType, argTypes[i]) {
			return false
		}
	}

	return true
}

// shouldRewriteIndex determines if an index expression should be rewritten
// STRICT RULE: Only rewrite if the base is a known struct type with magic methods
func (r *magicMethodRewriter) shouldRewriteIndex(expr Expr) bool {
	// Don't rewrite if we're inside a magic method definition
	if r.insideMagicMethod {
		return false
	}

	// STRICT: Only rewrite if we know the type has _getitem method
	return r.hasMagicMethodType(expr)
}

// shouldRewriteSlice determines if a slice expression should be rewritten
func (r *magicMethodRewriter) shouldRewriteSlice(sliceExpr *SliceExpr) bool {
	// Don't rewrite if we're inside a magic method definition
	if r.insideMagicMethod {
		return false
	}

	// STRICT: Only rewrite if we know the type has _getitem method
	return r.hasMagicMethodType(sliceExpr.X)
}

// hasMagicMethodType checks if an expression's type is a known struct with magic methods
// This is the ONLY way an index/slice expression will be rewritten
func (r *magicMethodRewriter) hasMagicMethodType(expr Expr) bool {
	typeName := r.tryInferStructTypeName(expr)
	if typeName == "" {
		return false
	}

	// Check exact match first
	_, exists := r.magicMethods[typeName]
	if exists {
		return true
	}

	// Go 允许值类型调用指针接收者的方法（自动取地址）
	// 所以如果我们有值类型，尝试查找指针类型的魔法方法
	if len(typeName) > 0 && typeName[0] != '*' {
		_, exists = r.magicMethods["*"+typeName]
		return exists
	}

	// 反之，如果我们有指针类型，也尝试查找值类型的魔法方法
	if len(typeName) > 0 && typeName[0] == '*' {
		_, exists = r.magicMethods[typeName[1:]]
		return exists
	}

	return false
}

// tryInferStructTypeName attempts to infer the struct type name from an expression
// Returns empty string if type cannot be determined or is not a struct
func (r *magicMethodRewriter) tryInferStructTypeName(expr Expr) string {
	switch e := expr.(type) {
	case *Name:
		// Variable reference - look it up in tracked variables
		if typeName, ok := r.varTypes[e.Value]; ok {
			return typeName
		}
		return ""

	case *ParenExpr:
		return r.tryInferStructTypeName(e.X)

	case *Operation:
		// Handle &Type{} or *var
		if e.Op == And && e.Y == nil {
			// Address-of: &Type{} or &var
			return r.tryInferStructTypeName(e.X)
		}
		if e.Op == Mul && e.Y == nil {
			// Dereference: *ptr
			return r.tryInferStructTypeName(e.X)
		}
		return ""

	case *CompositeLit:
		// Composite literal: Type{...}
		if e.Type != nil {
			return typeExprToString(e.Type)
		}
		return ""

	case *CallExpr:
		// Function call - check if it's a known constructor or make()
		if name, ok := e.Fun.(*Name); ok {
			// Special handling for make(Type, ...) - extract the type from first argument
			if name.Value == "make" && len(e.ArgList) > 0 {
				// make(Type, ...) - first arg is the type
				typeName := typeExprToString(e.ArgList[0])
				// Remove * prefix if present (though make doesn't work with pointers)
				if len(typeName) > 0 && typeName[0] == '*' {
					typeName = typeName[1:]
				}
				return typeName
			}

			// Check if this function returns a magic method type
			if returnType, exists := r.funcReturnTypes[name.Value]; exists {
				return returnType
			}
		}
		return ""

	case *SelectorExpr:
		// Field access like obj.field - we can't know the field's type
		return ""

	default:
		return ""
	}
}

// hasGetItemMethod checks if an expression's type has _getitem method
func (r *magicMethodRewriter) hasGetItemMethod(expr Expr) bool {
	typeName := r.inferTypeName(expr)
	if typeName == "" {
		return false
	}

	info, exists := r.magicMethods[typeName]
	return exists && len(info.GetItemMethods) > 0
}

// hasSetItemMethod checks if an expression's type has _setitem method
func (r *magicMethodRewriter) hasSetItemMethod(expr Expr) bool {
	typeName := r.inferTypeName(expr)
	if typeName == "" {
		return false
	}

	info, exists := r.magicMethods[typeName]
	return exists && len(info.SetItemMethods) > 0
}

// inferTypeName tries to infer the type name from an expression
func (r *magicMethodRewriter) inferTypeName(expr Expr) string {
	switch e := expr.(type) {
	case *Name:
		// Variable name - we'd need type info to resolve this
		// For now, return empty (will be handled by type checker later)
		return ""
	case *SelectorExpr:
		// Could be a field access or method call
		return ""
	case *CallExpr:
		// Function call - would need to know return type
		return ""
	case *ParenExpr:
		return r.inferTypeName(e.X)
	case *Operation:
		// Could be &Type{} or *var
		if e.Op == And && e.Y == nil {
			// Address-of operation
			return r.inferTypeName(e.X)
		}
		if e.Op == Mul && e.Y == nil {
			// Dereference operation
			return r.inferTypeName(e.X)
		}
		return ""
	case *CompositeLit:
		// Type literal like DataStore{}
		if e.Type != nil {
			return typeExprToString(e.Type)
		}
		return ""
	default:
		return ""
	}
}

// RewriteMethodDecorators rewrites decorated methods by wrapping their bodies
// with decorator function calls.
//
// Example transformation:
//
//	@logging
//	func (ds *DataStore) Get(key string, defaultValue string) string {
//	    if v, ok := ds.stringData[key]; ok { return v }
//	    return defaultValue
//	}
//
// becomes:
//
//	func (ds *DataStore) Get(key string, defaultValue string) string {
//	    _decorated := logging(func(key string, defaultValue string) string {
//	        if v, ok := ds.stringData[key]; ok { return v }
//	        return defaultValue
//	    })
//	    return _decorated(key, defaultValue)
//	}
func RewriteMethodDecorators(file *File) {
	for _, decl := range file.DeclList {
		if fn, ok := decl.(*FuncDecl); ok {
			if fn.Decorator != nil && fn.Recv != nil {
				rewriteDecoratedMethod(fn)
			}
		}
	}
}

func rewriteDecoratedMethod(fn *FuncDecl) {
	if fn.Body == nil || fn.Type == nil {
		return
	}

	pos := fn.Pos()
	decoratorName := fn.Decorator

	// At this point, default params haven't been rewritten yet,
	// so fn.Type.ParamList contains the original parameters
	// Create a function literal with the same signature (without receiver)
	funcLit := new(FuncLit)
	funcLit.SetPos(pos)
	funcLit.Type = &FuncType{
		ParamList:  fn.Type.ParamList,
		ResultList: fn.Type.ResultList,
	}
	funcLit.Type.SetPos(pos)
	funcLit.Body = fn.Body

	// Create decorator call: decoratorName(funcLit)
	decoratorCall := new(CallExpr)
	decoratorCall.SetPos(pos)
	decoratorCall.Fun = decoratorName
	decoratorCall.ArgList = []Expr{funcLit}

	// Create: _decorated := decoratorCall
	decoratedName := NewName(pos, "_decorated")
	decoratedRef := NewName(pos, "_decorated")
	decoratedAssign := &AssignStmt{
		Op:  Def,
		Lhs: decoratedName,
		Rhs: decoratorCall,
	}
	decoratedAssign.SetPos(pos)

	// Create argument list for calling _decorated: param1, param2, ...
	var callArgs []Expr
	for _, param := range fn.Type.ParamList {
		if param.Name != nil {
			argName := NewName(pos, param.Name.Value)
			callArgs = append(callArgs, argName)
		}
	}

	// Create: _decorated(param1, param2, ...)
	wrappedCall := new(CallExpr)
	wrappedCall.SetPos(pos)
	wrappedCall.Fun = decoratedRef
	wrappedCall.ArgList = callArgs

	// Build new method body
	newBody := new(BlockStmt)
	newBody.SetPos(pos)
	newBody.List = []Stmt{decoratedAssign}

	// Add return statement if function has return values
	if fn.Type.ResultList != nil && len(fn.Type.ResultList) > 0 {
		returnStmt := new(ReturnStmt)
		returnStmt.SetPos(pos)
		returnStmt.Results = wrappedCall
		newBody.List = append(newBody.List, returnStmt)
	} else {
		// No return value, just call the decorated function
		exprStmt := new(ExprStmt)
		exprStmt.SetPos(pos)
		exprStmt.X = wrappedCall
		newBody.List = append(newBody.List, exprStmt)
	}

	// Replace method body with wrapped version
	fn.Body = newBody
	fn.Decorator = nil // Clear decorator marker
}
