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

				// 提取类型参数名列表
				var tparamNames []string
				for _, tp := range td.TParamList {
					if tp.Name != nil {
						tparamNames = append(tparamNames, tp.Name.Value)
					}
				}

				isGeneric := len(tparamNames) > 0
				m := make(map[string]enumVariantInfo, len(enumType.VariantList))
				for i, v := range enumType.VariantList {
					if v != nil && v.Name != nil {
						m[v.Name.Value] = enumVariantInfo{
							tag:        i,
							hasPayload: v.Type != nil,
							payload:    v.Type,
							tparams:    tparamNames,
							isGeneric:  isGeneric,
						}
					}
				}
				r.enumVariants[td.Name.Value] = m

				// 2. 生成构造函数 (传入类型参数列表以支持泛型)
				ctors := r.generateEnumConstructors(td.Name, enumType, td.TParamList)
				newDecls = append(newDecls, ctors...)

				pos := td.Pos() // 获取位置信息

				// 分析所有变体，决定需要哪些字段
				// 对于泛型 enum，默认仍保留 _heap（兼容含指针/未知形状），但如果 types2 推导后发现
				// 存在“无指针 shape”的实例化，则为该 enum 增加 _stack 以支持栈上存储。
				var maxPureSize int
				var hasPointers bool
				if isGeneric {
					// 泛型 enum: heap 字段保守保留（只要存在 payload）
					hasPointers = hasAnyPayloadVariant(enumType.VariantList)
					// stack 大小来自 noder 第一遍 types2 的 shape hint（按 payload 无指针的最大 size）
					if h, ok := getEnumLoweringHint(td.Name.Value); ok {
						maxPureSize = h.MaxStack
						if h.NeedHeap {
							hasPointers = true
						}
					}
				} else {
					// 非泛型 enum: 分析变体类型决定存储策略
					maxPureSize = computeMaxPureValueSize(enumType.VariantList)
					hasPointers = hasAnyPointerVariant(enumType.VariantList)
				}

				// 字段 1: _tag int (总是需要)
				fields := []*Field{
					{
						Name: NewName(pos, "_tag"),
						Type: NewName(pos, "int"),
					},
				}

				// 字段 2: _stack [N]byte (如果有纯值变体且非泛型)
				if maxPureSize > 0 {
					stackField := &Field{
						Name: NewName(pos, "_stack"),
						Type: &ArrayType{
							Len:  &BasicLit{Value: strconv.Itoa(maxPureSize), Kind: IntLit},
							Elem: NewName(pos, "byte"),
						},
					}
					fields = append(fields, stackField)
				}

				// 字段 3: _heap unsafe.Pointer (如果有含指针变体或泛型)
				if hasPointers {
					heapField := &Field{
						Name: NewName(pos, "_heap"),
						Type: &SelectorExpr{
							X:   NewName(pos, "unsafe"),
							Sel: NewName(pos, "Pointer"),
						},
					}
					fields = append(fields, heapField)
				}

				// 替换 AST：Enum 变成了 Struct
				td.Type = &StructType{
					FieldList: fields,
				}

				// 标记需要引入 unsafe 包
				r.needsUnsafe = true
			}
		}
	}
	r.file.DeclList = newDecls
}

func (r *rewriter) generateEnumConstructors(typeName *Name, enum *EnumType, tparams []*Field) []Decl {
	var decls []Decl

	const (
		payloadAuto = iota
		payloadStack
		payloadHeap
	)

	emitCtor := func(v *EnumVariant, tag int, fnSuffix string, storage int) {
		T := func(n Node) Node {
			n.SetPos(v.Pos())
			return n
		}

		baseName := typeName.Value + "_" + v.Name.Value
		fnName := baseName + fnSuffix

		funcName := NewName(v.Pos(), fnName)
		fn := new(FuncDecl)
		T(fn)
		fn.Name = funcName

		// 复制类型参数列表以支持泛型
		fn.TParamList = tparams

		// --- 1. 构建参数列表 (支持多参数) ---
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

		// --- 2. 返回值：构建泛型实例化类型 ---
		var retType Expr
		if len(tparams) > 0 {
			var typeArgs []Expr
			for _, tp := range tparams {
				typeArgs = append(typeArgs, tp.Name)
			}
			if len(typeArgs) == 1 {
				retType = &IndexExpr{
					X:     NewName(v.Pos(), typeName.Value),
					Index: typeArgs[0],
				}
			} else {
				retType = &IndexExpr{
					X: NewName(v.Pos(), typeName.Value),
					Index: &ListExpr{
						ElemList: typeArgs,
					},
				}
			}
		} else {
			retType = NewName(v.Pos(), typeName.Value)
		}

		resField := &Field{Type: retType}
		T(resField)

		fn.Type = &FuncType{
			ParamList:  params,
			ResultList: []*Field{resField},
		}
		T(fn.Type)

		// --- 3. 生成函数体 ---
		fn.Body = r.generateConstructorBody(v.Pos(), typeName, tag, params, typeList, tparams, storage)

		decls = append(decls, fn)
	}

	for i, v := range enum.VariantList {
		// Unit variants: single constructor.
		if v.Type == nil {
			emitCtor(v, i, "", payloadAuto)
			continue
		}

		// Non-generic enums: single constructor (auto storage selection).
		if len(tparams) == 0 {
			emitCtor(v, i, "", payloadAuto)
			continue
		}

		// Generic payload variants: emit both stack/heap constructors.
		// Call-site lowering chooses based on types2-inferred variant payload layout.
		if h, ok := getEnumLoweringHint(typeName.Value); ok && h.MaxStack > 0 {
			emitCtor(v, i, "_stack", payloadStack)
		}
		emitCtor(v, i, "_heap", payloadHeap)
	}
	return decls
}

func (r *rewriter) generateConstructorBody(pos Pos, typeName *Name, tagVal int, params []*Field, types []Expr, tparams []*Field, storage int) *BlockStmt {
	T := func(n Node) Node { n.SetPos(pos); return n }

	block := new(BlockStmt)
	block.SetPos(pos)

	// 1. var ret TypeName[T1, T2, ...] (如果是泛型)
	retName := NewName(pos, "ret")

	var retTypeExpr Expr
	if len(tparams) > 0 {
		// 泛型 enum: var ret EnumName[T1, T2, ...]
		var typeArgs []Expr
		for _, tp := range tparams {
			typeArgs = append(typeArgs, tp.Name)
		}

		if len(typeArgs) == 1 {
			retTypeExpr = &IndexExpr{
				X:     NewName(pos, typeName.Value),
				Index: typeArgs[0],
			}
		} else {
			retTypeExpr = &IndexExpr{
				X: NewName(pos, typeName.Value),
				Index: &ListExpr{
					ElemList: typeArgs,
				},
			}
		}
	} else {
		// 非泛型 enum
		retTypeExpr = NewName(pos, typeName.Value)
	}

	varDecl := &VarDecl{
		NameList: []*Name{retName},
		Type:     retTypeExpr,
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

	// 3. Payload 存储：根据类型选择 _stack 或 _heap
	if len(params) == 0 {
		// Unit variant: 只有 tag，无 payload
		retStmt := &ReturnStmt{Results: retName}
		T(retStmt)
		block.List = append(block.List, retStmt)
		return block
	}

	const (
		payloadAuto = iota
		payloadStack
		payloadHeap
	)

	// 判断是否包含指针或是否是泛型 enum
	isGeneric := len(tparams) > 0
	hasPointer := false
	for _, t := range types {
		if containsPointer(t) {
			hasPointer = true
			break
		}
	}

	useHeap := false
	switch storage {
	case payloadHeap:
		useHeap = true
	case payloadStack:
		useHeap = false
	default:
		useHeap = isGeneric || hasPointer
	}

	if !useHeap {
		// === 路径 A: 写入 _stack ===
		// 使用一次性 typed assignment，避免依赖“估算偏移”。

		// &ret._stack[0]
		stackSel := &SelectorExpr{X: retName, Sel: NewName(pos, "_stack")}
		T(stackSel)
		zeroLit := &BasicLit{Value: "0", Kind: IntLit}
		T(zeroLit)
		indexExpr := &IndexExpr{X: stackSel, Index: zeroLit}
		T(indexExpr)
		addrStack := &Operation{Op: And, X: indexExpr}
		T(addrStack)

		ptrStack := &CallExpr{
			Fun:     &SelectorExpr{X: NewName(pos, "unsafe"), Sel: NewName(pos, "Pointer")},
			ArgList: []Expr{addrStack},
		}
		T(ptrStack)

		var payloadType Expr
		var payloadValue Expr
		if len(types) == 1 {
			payloadType = types[0]
			payloadValue = params[0].Name
		} else {
			structFields := make([]*Field, len(types))
			for idx, t := range types {
				structFields[idx] = &Field{Name: NewName(pos, "_"+strconv.Itoa(idx)), Type: t}
			}
			payloadType = &StructType{FieldList: structFields}
			T(payloadType)

			compLit := &CompositeLit{Type: payloadType}
			T(compLit)
			for _, p := range params {
				compLit.ElemList = append(compLit.ElemList, p.Name)
			}
			payloadValue = compLit
		}

		ptrT := &Operation{Op: Mul, X: payloadType}
		T(ptrT)
		castT := &CallExpr{
			Fun:     &ParenExpr{X: ptrT},
			ArgList: []Expr{ptrStack},
		}
		T(castT)
		derefT := &Operation{Op: Mul, X: castT}
		T(derefT)

		assign := &AssignStmt{Op: 0, Lhs: derefT, Rhs: payloadValue}
		T(assign)
		block.List = append(block.List, assign)

	} else {
		// === 路径 B: 含指针变体 → 分配到堆，写入 _heap ===

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

		// ret._heap = unsafe.Pointer(...)
		toPtr := &CallExpr{
			Fun:     &SelectorExpr{X: NewName(pos, "unsafe"), Sel: NewName(pos, "Pointer")},
			ArgList: []Expr{addrExpr},
		}
		T(toPtr)

		heapAssign := &AssignStmt{
			Op:  0,
			Lhs: &SelectorExpr{X: retName, Sel: NewName(pos, "_heap")},
			Rhs: toPtr,
		}
		T(heapAssign)
		block.List = append(block.List, heapAssign)
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

// containsPointer 递归判断类型是否包含指针
func containsPointer(typ Expr) bool {
	if typ == nil {
		return false
	}

	switch t := typ.(type) {
	case *Name:
		switch t.Value {
		// 纯值类型
		case "int", "int8", "int16", "int32", "int64",
			"uint", "uint8", "uint16", "uint32", "uint64",
			"float32", "float64", "bool", "byte", "rune", "uintptr":
			return false
		// 含指针类型
		case "string": // string 是 {ptr, len}
			return true
		default:
			// 自定义类型，保守假设含指针（除非做全局分析）
			return true
		}

	case *Operation: // *T (显式指针)
		if t.Op == Mul && t.Y == nil {
			return true
		}

	case *ArrayType: // [N]T - 递归检查元素
		return containsPointer(t.Elem)

	case *SliceType, *MapType, *ChanType, *InterfaceType, *FuncType:
		return true

	case *StructType:
		// 匿名 struct，检查所有字段
		for _, f := range t.FieldList {
			if containsPointer(f.Type) {
				return true
			}
		}
		return false

	case *ListExpr:
		// Tuple: 任何一个元素含指针则整体含指针
		for _, elem := range t.ElemList {
			if containsPointer(elem) {
				return true
			}
		}
		return false
	}

	return true // 默认保守假设含指针
}

// variantContainsPointer 判断变体的 payload 是否包含指针
func variantContainsPointer(v *EnumVariant) bool {
	if v == nil || v.Type == nil {
		return false // unit variant
	}
	types := unpackTypeList(v.Type)
	for _, t := range types {
		if containsPointer(t) {
			return true
		}
	}
	return false
}

// estimateTypeSize 粗略估算类型大小（字节）
func estimateTypeSize(typ Expr) int {
	if typ == nil {
		return 0
	}

	switch t := typ.(type) {
	case *Name:
		switch t.Value {
		case "bool", "int8", "uint8", "byte":
			return 1
		case "int16", "uint16":
			return 2
		case "int32", "uint32", "float32", "rune":
			return 4
		case "int", "int64", "uint", "uint64", "float64", "uintptr":
			return 8
		case "complex64":
			return 8
		case "complex128":
			return 16
		default:
			return 8 // 保守估计
		}

	case *ArrayType:
		elemSize := estimateTypeSize(t.Elem)
		if lit, ok := t.Len.(*BasicLit); ok {
			if n, err := strconv.Atoi(lit.Value); err == nil {
				return elemSize * n
			}
		}
		return elemSize * 4 // 默认假设小数组

	case *StructType:
		total := 0
		for _, f := range t.FieldList {
			total += estimateTypeSize(f.Type)
		}
		// 简单对齐到 8 字节
		if total%8 != 0 {
			total = ((total / 8) + 1) * 8
		}
		return total

	case *ListExpr:
		// Tuple: 累加所有元素
		total := 0
		for _, elem := range t.ElemList {
			total += estimateTypeSize(elem)
		}
		return total
	}

	return 8 // 默认
}

// computeMaxPureValueSize 计算所有纯值变体的最大 payload 大小
func computeMaxPureValueSize(variants []*EnumVariant) int {
	maxSize := 0
	for _, v := range variants {
		if v == nil || v.Type == nil {
			continue
		}
		if !variantContainsPointer(v) {
			size := estimateTypeSize(v.Type)
			if size > maxSize {
				maxSize = size
			}
		}
	}
	return maxSize
}

// hasAnyPointerVariant 判断是否有任何变体包含指针
func hasAnyPointerVariant(variants []*EnumVariant) bool {
	for _, v := range variants {
		if variantContainsPointer(v) {
			return true
		}
	}
	return false
}

// hasAnyPayloadVariant 判断是否有任何变体有 payload
func hasAnyPayloadVariant(variants []*EnumVariant) bool {
	for _, v := range variants {
		if v != nil && v.Type != nil {
			return true
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
