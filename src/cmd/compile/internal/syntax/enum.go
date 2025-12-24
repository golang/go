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

				// 2. 生成构造函数 (保持不变，但 generateConstructorBody 内部实现需要对应修改!)
				ctors := r.generateEnumConstructors(td.Name, enumType)
				newDecls = append(newDecls, ctors...)

				pos := td.Pos() // 获取位置信息

				// 字段 1: _tag int
				tagField := &Field{
					Name: NewName(pos, "_tag"),
					Type: NewName(pos, "int"),
				}

				// 字段 2: _ptr unsafe.Pointer
				ptrField := &Field{
					Name: NewName(pos, "_ptr"),
					Type: &SelectorExpr{
						X:   NewName(pos, "unsafe"),
						Sel: NewName(pos, "Pointer"),
					},
				}

				// 字段 3: _scalar uint64
				// 用来存放 int, float, bool 等非指针数据
				scalarField := &Field{
					Name: NewName(pos, "_scalar"),
					Type: NewName(pos, "uint64"), // 足够存 float64 或 int64
				}

				// 替换 AST：Enum 变成了 Struct
				td.Type = &StructType{
					FieldList: []*Field{tagField, ptrField, scalarField},
				}

				// 标记需要引入 unsafe 包 (因为 struct 定义里用了 unsafe.Pointer)
				r.needsUnsafe = true
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

		// --- 1. 构建参数列表 (支持多参数) ---
		// 使用 unpackTypeList 将 int, int 拆开
		typeList := unpackTypeList(v.Type)

		var params []*Field
		for idx, t := range typeList {
			argName := "v"
			if len(typeList) > 1 {
				argName = "v" + strconv.Itoa(idx)
			}
			paramField := &Field{
				Name: NewName(v.Pos(), argName),
				Type: t,
			}
			T(paramField)
			params = append(params, paramField)
		}

		// --- 2. 返回值 ---
		retType := NewName(v.Pos(), typeName.Value)
		resField := &Field{Type: retType}
		T(resField)

		fn.Type = &FuncType{
			ParamList:  params,
			ResultList: []*Field{resField},
		}
		T(fn.Type)

		// --- 3. 生成函数体 ---
		// 关键修复：传入 params slice 和 typeList slice
		fn.Body = r.generateConstructorBody(v.Pos(), typeName, i, params, typeList)

		decls = append(decls, fn)
	}
	return decls
}

func (r *rewriter) generateConstructorBody(pos Pos, typeName *Name, tagVal int, params []*Field, types []Expr) *BlockStmt {
	T := func(n Node) Node { n.SetPos(pos); return n }

	block := new(BlockStmt)
	block.SetPos(pos)

	// 1. var ret TypeName
	retName := NewName(pos, "ret")
	varDecl := &VarDecl{
		NameList: []*Name{retName},
		Type:     NewName(pos, typeName.Value),
	}
	T(varDecl)
	block.List = append(block.List, &DeclStmt{DeclList: []Decl{varDecl}})

	// 2. ret._tag = tagVal
	tagAssign := &AssignStmt{
		Op:  0,
		Lhs: &SelectorExpr{X: retName, Sel: NewName(pos, "_tag")},
		Rhs: &BasicLit{Value: strconv.Itoa(tagVal), Kind: IntLit},
	}
	T(tagAssign)
	block.List = append(block.List, tagAssign)

	// 3. Payload 存储分流
	if len(params) == 1 && isOptimizableScalar(types[0]) {
		// === 路径 A: 标量优化 (int, bool, float...) ===
		// ret._scalar = *(*uint64)(unsafe.Pointer(&v))
		// 这样可以处理 float 到 uint64 的 bit-cast，也可以处理 bool

		// &v
		addrParam := &Operation{Op: And, X: params[0].Name}

		// unsafe.Pointer(&v)
		ptrParam := &CallExpr{
			Fun:     &SelectorExpr{X: NewName(pos, "unsafe"), Sel: NewName(pos, "Pointer")},
			ArgList: []Expr{addrParam},
		}

		// (*uint64)(ptr)
		castPtr := &CallExpr{
			Fun:     &ParenExpr{X: &Operation{Op: Mul, X: NewName(pos, "uint64")}}, // *uint64
			ArgList: []Expr{ptrParam},
		}

		// *castPtr
		valUint64 := &Operation{Op: Mul, X: castPtr}

		// ret._scalar = ...
		scalarAssign := &AssignStmt{
			Op:  0,
			Lhs: &SelectorExpr{X: retName, Sel: NewName(pos, "_scalar")},
			Rhs: valUint64,
		}
		T(scalarAssign)
		block.List = append(block.List, scalarAssign)

	} else if len(params) > 0 {
		// === 路径 B: 通用路径 (指针 或 Tuple) ===
		// 打包成匿名结构体指针: &struct{...}{...} -> _ptr

		// 构造匿名结构体类型
		structFields := make([]*Field, len(types))
		for idx, t := range types {
			structFields[idx] = &Field{Name: NewName(pos, "_"+strconv.Itoa(idx)), Type: t}
		}
		anonStructType := &StructType{FieldList: structFields}
		T(anonStructType)

		// 构造 CompositeLit
		compLit := &CompositeLit{Type: anonStructType}
		T(compLit)
		for _, p := range params {
			compLit.ElemList = append(compLit.ElemList, p.Name)
		}

		// 取地址 &Literal -> 堆分配
		addrExpr := &Operation{Op: And, X: compLit}
		T(addrExpr)

		// ret._ptr = unsafe.Pointer(...)
		toPtr := &CallExpr{
			Fun:     &SelectorExpr{X: NewName(pos, "unsafe"), Sel: NewName(pos, "Pointer")},
			ArgList: []Expr{addrExpr},
		}
		T(toPtr)

		ptrAssign := &AssignStmt{
			Op:  0,
			Lhs: &SelectorExpr{X: retName, Sel: NewName(pos, "_ptr")},
			Rhs: toPtr,
		}
		T(ptrAssign)
		block.List = append(block.List, ptrAssign)
	}

	// return ret
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

func (r *rewriter) isPointerType(typ Expr) bool {
	switch t := typ.(type) {
	case *Operation: // *T (指针)
		return t.Op == Mul
	case *MapType, *ChanType, *FuncType, *InterfaceType:
		return true
	case *Name:
		// struct/slice/string to scalar
		return false
	}
	return false
}

func isOptimizableScalar(t Expr) bool {
	if name, ok := t.(*Name); ok {
		switch name.Value {
		// 8字节及以下的数值类型，且不含指针
		case "int", "int8", "int16", "int32", "int64":
			return true
		case "uint", "uint8", "uint16", "uint32", "uint64", "byte", "rune":
			return true
		case "bool":
			return true
		case "float32", "float64":
			return true
		case "uintptr":
			return true
			// 注意：string 和 complex128 大于 8 字节，必须走指针路径
		}
	}
	return false
}

func unpackTypeList(x Expr) []Expr {
	if x == nil {
		return nil
	}
	if l, ok := x.(*ListExpr); ok {
		return l.ElemList
	}
	return []Expr{x}
}
