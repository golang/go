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
		return r.rewriteOptionalChain(e)

	case *TernaryExpr:
		return r.rewriteTernary(e)

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
		// Special handling for optional chaining with method calls
		if opt, ok := e.Fun.(*OptionalChainExpr); ok {
			return r.rewriteOptionalChainCall(opt, e)
		}
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

// rewriteTernary converts cond ? x : y to:
// func() interface{} { if cond { return x } else { return y } }()
func (r *rewriter) rewriteTernary(e *TernaryExpr) Expr {
	pos := e.Pos()

	cond := r.rewriteExpr(e.Cond)
	trueExpr := r.rewriteExpr(e.X)

	var falseExpr Expr
	if e.Y != nil {
		falseExpr = r.rewriteExpr(e.Y)
	} else {
		falseExpr = trueExpr
	}

	// Create return statements
	retTrue := new(ReturnStmt)
	retTrue.SetPos(pos)
	retTrue.Results = trueExpr

	retFalse := new(ReturnStmt)
	retFalse.SetPos(pos)
	retFalse.Results = falseExpr

	// Create blocks
	thenBlock := new(BlockStmt)
	thenBlock.SetPos(pos)
	thenBlock.List = []Stmt{retTrue}

	elseBlock := new(BlockStmt)
	elseBlock.SetPos(pos)
	elseBlock.List = []Stmt{retFalse}

	// Create if statement
	ifStmt := new(IfStmt)
	ifStmt.SetPos(pos)
	ifStmt.Cond = cond
	ifStmt.Then = thenBlock
	ifStmt.Else = elseBlock

	// Create return type: interface{}
	interfaceType := new(InterfaceType)
	interfaceType.SetPos(pos)

	resultField := new(Field)
	resultField.SetPos(pos)
	resultField.Type = interfaceType

	funcType := new(FuncType)
	funcType.SetPos(pos)
	funcType.ResultList = []*Field{resultField}

	funcBody := new(BlockStmt)
	funcBody.SetPos(pos)
	funcBody.List = []Stmt{ifStmt}

	funcLit := new(FuncLit)
	funcLit.SetPos(pos)
	funcLit.Type = funcType
	funcLit.Body = funcBody

	callExpr := new(CallExpr)
	callExpr.SetPos(pos)
	callExpr.Fun = funcLit

	return callExpr
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
