package syntax

import (
	"strconv"
)

func (r *rewriter) rewriteEnums() {
	var newDecls []Decl
	for _, decl := range r.file.DeclList {
		newDecls = append(newDecls, decl)

		// 检查是否是 TypeDecl -> EnumType
		if td, ok := decl.(*TypeDecl); ok {
			if enumType, ok := td.Type.(*EnumType); ok {
				// Record enum variants for later expression rewriting (EnumName.Variant(...)).
				if r.enumVariants == nil {
					r.enumVariants = make(map[string]map[string]enumVariantInfo)
				}
				m := make(map[string]enumVariantInfo, len(enumType.VariantList))
				for i, v := range enumType.VariantList {
					if v != nil && v.Name != nil {
						m[v.Name.Value] = enumVariantInfo{
							tag:        i,
							hasPayload: v.Type != nil,
							payload:    v.Type,
						}
					}
				}
				r.enumVariants[td.Name.Value] = m

				// 为该 Enum 生成所有变体的构造函数
				ctors := r.generateEnumConstructors(td.Name, enumType)
				newDecls = append(newDecls, ctors...)

				if len(ctors) > 0 {
					r.needsUnsafe = true
				}
			}
		}
	}
	r.file.DeclList = newDecls
}

func (r *rewriter) generateEnumConstructors(typeName *Name, enum *EnumType) []Decl {
	var decls []Decl

	for i, v := range enum.VariantList {
		T := func(n Node) Node {
			n.SetPos(v.Pos())
			return n
		}

		funcName := NewName(v.Pos(), typeName.Value+"_"+v.Name.Value)

		fn := new(FuncDecl)
		T(fn)
		fn.Name = funcName

		// 1. 构建参数列表
		var params []*Field
		var paramName *Name

		if v.Type != nil {
			paramName = NewName(v.Pos(), "v")

			paramField := &Field{
				Name: paramName,
				Type: v.Type,
			}
			T(paramField)

			params = []*Field{paramField}
		}

		// 2. 构建返回值列表 (返回 Enum 类型)
		retType := NewName(v.Pos(), typeName.Value)
		resField := &Field{
			Type: retType,
		}
		T(resField)

		results := []*Field{resField}

		// 3. 构建函数类型 (FuncType)
		ft := &FuncType{
			ParamList:  params,
			ResultList: results,
		}
		T(ft)
		fn.Type = ft

		fn.Body = r.generateConstructorBody(v.Pos(), typeName, i, paramName, v.Type)

		decls = append(decls, fn)
	}
	return decls
}

func (r *rewriter) generateConstructorBody(pos Pos, typeName *Name, tagVal int, paramName *Name, paramType Expr) *BlockStmt {
	T := func(n Node) Node {
		n.SetPos(pos)
		return n
	}

	block := new(BlockStmt)
	block.SetPos(pos)

	retName := NewName(pos, "ret")

	newCall := &CallExpr{
		Fun:     NewName(pos, "new"),
		ArgList: []Expr{NewName(pos, typeName.Value)},
	}
	T(newCall)

	deref := &Operation{
		Op: Mul,
		X:  newCall,
	}
	T(deref)

	assignRet := &AssignStmt{
		Op:  Def,
		Lhs: retName,
		Rhs: deref,
	}
	T(assignRet)
	block.List = append(block.List, assignRet)

	// Helper: create `unsafe.Pointer(&ret)`
	unsafePtr := func() Expr {
		// &ret
		addrRet := &Operation{Op: And, X: retName}
		T(addrRet)

		// unsafe.Pointer
		sel := &SelectorExpr{X: NewName(pos, "unsafe"), Sel: NewName(pos, "Pointer")}
		T(sel)

		// unsafe.Pointer(&ret)
		call := &CallExpr{
			Fun:     sel,
			ArgList: []Expr{addrRet},
		}
		T(call)
		return call
	}

	// 2. Set Tag: *(*int)(unsafe.Pointer(&ret)) = tagVal
	{
		// (*int)
		ptrInt := &Operation{Op: Mul, X: NewName(pos, "int")}
		T(ptrInt)

		// (*int)(...)
		paren := &ParenExpr{X: ptrInt}
		T(paren)

		// Cast Call
		cast := &CallExpr{Fun: paren, ArgList: []Expr{unsafePtr()}}
		T(cast)

		// *...
		deref := &Operation{Op: Mul, X: cast}
		T(deref)

		// Lit: tagVal
		tagLit := &BasicLit{Value: strconv.Itoa(tagVal), Kind: IntLit}
		T(tagLit)

		// = tagVal
		assign := &AssignStmt{
			Op:  0, // =
			Lhs: deref,
			Rhs: tagLit,
		}
		T(assign)
		block.List = append(block.List, assign)
	}

	// 3. Set Payload (if exists)
	if paramName != nil {
		// Construct: unsafe.Sizeof(int(0))

		// int(0)
		zeroLit := &BasicLit{Value: "0", Kind: IntLit}
		T(zeroLit) // <--- 别漏了

		intCall := &CallExpr{Fun: NewName(pos, "int"), ArgList: []Expr{zeroLit}}
		T(intCall)

		// unsafe.Sizeof
		selSizeof := &SelectorExpr{X: NewName(pos, "unsafe"), Sel: NewName(pos, "Sizeof")}
		T(selSizeof)

		sizeOfCall := &CallExpr{
			Fun:     selSizeof,
			ArgList: []Expr{intCall},
		}
		T(sizeOfCall)

		// Construct: unsafe.Add(ptr, offset)
		selAdd := &SelectorExpr{X: NewName(pos, "unsafe"), Sel: NewName(pos, "Add")}
		T(selAdd)

		unsafeAdd := &CallExpr{
			Fun:     selAdd,
			ArgList: []Expr{unsafePtr(), sizeOfCall},
		}
		T(unsafeAdd)

		// Construct: (*PayloadType)(...)
		ptrPayload := &Operation{Op: Mul, X: paramType} // paramType 应该已有 Pos，但保险起见...
		// 注意：paramType 是传进来的，最好不要修改它的 Pos，或者它本身就是有 Pos 的。
		// 这里我们要给新创建的 Operation 加 Pos
		T(ptrPayload)

		parenPayload := &ParenExpr{X: ptrPayload}
		T(parenPayload)

		cast := &CallExpr{Fun: parenPayload, ArgList: []Expr{unsafeAdd}}
		T(cast)

		// *...
		derefPayload := &Operation{Op: Mul, X: cast}
		T(derefPayload)

		// = v
		assign := &AssignStmt{
			Op:  0,
			Lhs: derefPayload,
			Rhs: paramName,
		}
		T(assign)
		block.List = append(block.List, assign)
	}

	// 4. return ret
	retStmt := &ReturnStmt{Results: retName}
	T(retStmt)
	block.List = append(block.List, retStmt)

	return block
}

func (r *rewriter) addUnsafeImport() {
	r.addImport("unsafe")
}

func (r *rewriter) addImport(pkgName string) {
	if r.file == nil {
		return
	}
	// Check if already imported
	for _, decl := range r.file.DeclList {
		if imp, ok := decl.(*ImportDecl); ok {
			if imp.Path != nil {
				path := imp.Path.Value
				if path == `"`+pkgName+`"` || path == "`"+pkgName+"`" {
					return // Already imported
				}
			}
		}
	}

	// Find position
	pos := r.file.Pos()
	if len(r.file.DeclList) > 0 {
		pos = r.file.DeclList[0].Pos()
	}

	// Create import
	imp := &ImportDecl{
		Path: &BasicLit{Value: `"` + pkgName + `"`, Kind: StringLit},
	}
	imp.SetPos(pos)

	// Prepend to DeclList
	r.file.DeclList = append([]Decl{imp}, r.file.DeclList...)
}
