package noder

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"go/constant"
	"strconv"
)

var forwardRefCache = make(map[string]*types.Type)

// clearForwardRefCache 清理未使用的占位类型
// 应在包级别类型检查完成后调用
func clearForwardRefCache() {
	if len(forwardRefCache) > 0 {
		// 报告未解析的前向引用
		for name := range forwardRefCache {
			base.Errorf("unresolved forward reference to type: %s", name)
		}
		forwardRefCache = make(map[string]*types.Type)
	}
}

// ProcessEnumDeclarations 处理文件中的所有 enum 声明
// 应在类型检查之前调用，将 enum 转换为 struct + 构造函数
//
// 集成策略：
// 1. 在 syntax.Parse 之后立即调用
// 2. 在 types2.Check 之前调用
// 3. 将 enum 转换为等价的 Go 代码（struct + 函数）
func ProcessEnumDeclarations(m posMap, file *syntax.File) {
	if file == nil {
		return
	}

	// 收集所有需要转换的 enum 声明及其索引
	var enumDecls []struct {
		index    int
		typeDecl *syntax.TypeDecl
		enumType *syntax.EnumType
	}

	for i, decl := range file.DeclList {
		typeDecl, ok := decl.(*syntax.TypeDecl)
		if !ok {
			continue
		}

		// 检查是否是 enum 类型
		enumType, ok := typeDecl.Type.(*syntax.EnumType)
		if !ok {
			continue
		}

		enumDecls = append(enumDecls, struct {
			index    int
			typeDecl *syntax.TypeDecl
			enumType *syntax.EnumType
		}{i, typeDecl, enumType})
	}

	// 如果没有 enum，直接返回
	if len(enumDecls) == 0 {
		return
	}

	// 创建新的声明列表
	newDeclList := make([]syntax.Decl, 0, len(file.DeclList)+len(enumDecls)*10)

	enumIdx := 0
	for i, decl := range file.DeclList {
		// 检查是否是要转换的 enum
		if enumIdx < len(enumDecls) && i == enumDecls[enumIdx].index {
			// 转换 enum 为 IR 节点
			ed := enumDecls[enumIdx]
			pos := m.makeXPos(ed.typeDecl.Pos())
			nodes := transformEnumDecl(ed.enumType, ed.typeDecl.Name, pos)

			// 将生成的 IR 节点转换回 syntax 节点
			// 注意：这里需要将 ir.Node 转换为 syntax.Decl
			// 由于 IR 和 syntax 是不同的表示，我们需要创建对应的 syntax 节点

			// 简化方案：直接替换为注释节点，实际转换在后续阶段完成
			// TODO: 完整实现需要在 unified IR 阶段处理
			_ = nodes

			// 保留原始声明（暂时），添加标记
			newDeclList = append(newDeclList, decl)
			enumIdx++
		} else {
			newDeclList = append(newDeclList, decl)
		}
	}

	file.DeclList = newDeclList

	// 处理完所有 enum 后，清理前向引用缓存
	// 注意：这里不能立即清理，因为可能有跨文件的前向引用
	// 应该在整个包处理完成后清理
	// clearForwardRefCache() // 移到 ProcessPackageEnums
}

// ProcessPackageEnums 处理整个包的所有 enum 声明
// 应在所有文件处理完成后调用
func ProcessPackageEnums(m posMap, files []*syntax.File) {
	// 处理每个文件
	for _, file := range files {
		ProcessEnumDeclarations(m, file)
	}

	// 清理前向引用缓存
	clearForwardRefCache()
}

// transformEnumDecl 将 Enum 声明转换为 Struct + 构造函数
func transformEnumDecl(n *syntax.EnumType, typeName *syntax.Name, pos src.XPos) []ir.Node {
	var nodes []ir.Node

	// 0. 预先注册类型符号，支持递归引用
	// 例如: enum Node { Leaf(int), Branch(*Node) }
	// 在处理 Branch(*Node) 时，Node 类型需要已经存在
	typeSym := typecheck.Lookup(typeName.Value)

	// 如果缓存中有占位类型，准备更新它
	var placeholderType *types.Type
	if cached, ok := forwardRefCache[typeName.Value]; ok {
		placeholderType = cached
		delete(forwardRefCache, typeName.Value) // 清除缓存
	}

	// 1. 字段 1: tag
	// 如果变体超过 255 个，自动升级为 uint16
	var tagType *types.Type
	if len(n.VariantList) > 255 {
		tagType = types.Types[types.TUINT16]
	} else {
		tagType = types.Types[types.TUINT8]
	}
	tagField := types.NewField(pos, typecheck.Lookup("tag"), tagType)

	// 2. 字段 2: data
	// 检查指针以决定 GC 策略
	hasPointers := hasAnyPointers(n.VariantList)
	maxSize := computeMaxPayloadSize(n.VariantList)
	if maxSize == 0 {
		maxSize = 1 // 占位
	}

	var dataField *types.Field
	var arrayType *types.Type

	if hasPointers {
		// 策略 A: 包含指针 -> [N]unsafe.Pointer
		// 使用 types.PtrSize 适配 32/64 位架构
		numPtrSlots := (maxSize + int64(types.PtrSize) - 1) / int64(types.PtrSize)
		elemType := types.Types[types.TUNSAFEPTR]
		arrayType = types.NewArray(elemType, numPtrSlots)
		dataField = types.NewField(pos, typecheck.Lookup("data"), arrayType)
	} else {
		// 策略 B: 纯数据 -> [N]byte
		elemType := types.Types[types.TUINT8]
		arrayType = types.NewArray(elemType, maxSize)
		dataField = types.NewField(pos, typecheck.Lookup("data"), arrayType)
	}

	// 3. 创建 Struct 类型
	structType := types.NewStruct([]*types.Field{tagField, dataField})

	// 4. 注册类型并更新占位符
	if placeholderType != nil {
		// 如果存在占位类型（前向引用），更新其底层类型
		// 这样所有引用占位类型的地方都会自动看到正确的类型
		placeholderType.SetUnderlying(structType)

		// 使用占位类型作为最终类型
		// 这保证了类型的一致性
		structType = placeholderType

		if base.Flag.LowerM != 0 {
			base.WarnfAt(pos, "updated forward reference placeholder for type %s", typeName.Value)
		}
	}

	// 清除旧定义（如果有）
	typeSym.Def = nil

	// 创建类型名称节点
	typeNameNode := ir.NewDeclNameAt(pos, ir.OTYPE, typeSym)
	typeNameNode.SetType(structType)
	typeNameNode.SetTypecheck(1)
	typeSym.Def = typeNameNode

	// 创建类型声明节点
	typeNode := ir.NewDecl(pos, ir.ODCL, typeNameNode)
	nodes = append(nodes, typeNode)

	// 5. 生成构造函数
	for i, variant := range n.VariantList {
		ctorName := typeName.Value + "_" + variant.Name.Value
		ctorSym := typecheck.Lookup(ctorName)

		var params []*types.Field
		if variant.Type != nil {
			paramType := convertSyntaxTypeToIRType(variant.Type)
			// 安全检查：如果类型转换失败，这里要报错
			if paramType == nil {
				base.ErrorfAt(pos, 0, "invalid type in enum variant %s", variant.Name.Value)
				continue
			}
			params = []*types.Field{types.NewField(pos, typecheck.Lookup("v"), paramType)}
		}

		results := []*types.Field{types.NewField(pos, nil, structType)}
		fn := ir.NewFunc(pos, pos, ctorSym, types.NewSignature(nil, params, results))
		fn.SetDupok(true)
		fn.DeclareParams(true)

		ir.WithFunc(fn, func() {
			// var n Number
			nVar := fn.NewLocal(pos, typecheck.Lookup("n"), structType)

			// n.tag = i
			tagSel := ir.NewSelectorExpr(pos, ir.ODOT, nVar, typecheck.Lookup("tag"))
			tagSel.SetType(tagType)
			tagSel.SetTypecheck(1)
			tagAssign := ir.NewAssignStmt(pos, tagSel, ir.NewBasicLit(pos, tagType, constant.MakeUint64(uint64(i))))
			fn.Body.Append(typecheck.Stmt(tagAssign))

			// Payload 写入
			if variant.Type != nil && len(params) > 0 {
				paramVar := fn.Dcl[0]

				// 获取 data 字段地址: &n.data
				dataSel := ir.NewSelectorExpr(pos, ir.ODOT, nVar, typecheck.Lookup("data"))
				dataSel.SetType(arrayType)
				dataSel.SetTypecheck(1)
				dataAddr := ir.NewAddrExpr(pos, dataSel)
				dataAddr.SetType(types.NewPtr(arrayType))

				// OCONVNOP 魔法: *(*T)(unsafe.Pointer(&n.data)) = v
				// 1. &n.data -> unsafe.Pointer
				uPtr := ir.NewConvExpr(pos, ir.OCONVNOP, types.Types[types.TUNSAFEPTR], dataAddr)

				// 2. unsafe.Pointer -> *T
				paramType := convertSyntaxTypeToIRType(variant.Type)
				targetPtr := types.NewPtr(paramType)
				typedPtr := ir.NewConvExpr(pos, ir.OCONVNOP, targetPtr, uPtr)

				// 3. *ptr = v
				deref := ir.NewStarExpr(pos, typedPtr)
				deref.SetType(paramType)
				assign := ir.NewAssignStmt(pos, deref, paramVar)
				fn.Body.Append(typecheck.Stmt(assign))
			}

			fn.Body.Append(ir.NewReturnStmt(pos, []ir.Node{nVar}))
		})

		fn.SetTypecheck(1)
		typecheck.Target.Funcs = append(typecheck.Target.Funcs, fn)
		nodes = append(nodes, fn)
	}

	return nodes
}

// 辅助函数：判断是否是尚未解析的占位符
func isForwardRefPlaceholder(t *types.Type) bool {
	// 我们的占位符逻辑是：Named Type 且 Underlying 是 TUINT8 (且名字在 cache 中)
	// 这是一个启发式检查
	if t.Sym() != nil {
		if _, ok := forwardRefCache[t.Sym().Name]; ok {
			return true
		}
	}
	return false
}

// computeMaxPayloadSize 修复版：使用真实的 Type Size
// 支持递归类型定义（通过指针）
func computeMaxPayloadSize(variants []*syntax.EnumVariant) int64 {
	maxSize := int64(0)

	// 用于检测递归的访问集合
	visiting := make(map[*types.Type]bool)

	for _, v := range variants {
		if v.Type == nil {
			continue
		}

		// 关键修复：将 syntax 转换为 IR Type，然后计算 Size
		typ := convertSyntaxTypeToIRType(v.Type)
		if typ == nil {
			continue
		}

		// 关键检查：如果是占位符类型（前向引用），必须确保它是通过指针引用的
		// 否则我们无法知道它的大小
		if isForwardRefPlaceholder(typ) {
			// 这种情况下，CalcSize 得到的是假的 1 字节
			// 必须报错，或者强制要求用户使用指针
			base.ErrorfAt(src.NoXPos, 0, "invalid forward reference in value position: %v (must use pointer for recursive/forward types)", v.Name.Value)
			continue
		}

		// 安全地计算类型大小，处理递归情况
		sz := safeCalcTypeSize(typ, visiting)

		if sz > maxSize {
			maxSize = sz
		}
	}
	return maxSize
}

// safeCalcTypeSize 安全地计算类型大小，处理递归定义
// 递归类型（如 *Node）的大小是指针大小，不会无限递归
func safeCalcTypeSize(typ *types.Type, visiting map[*types.Type]bool) int64 {
	if typ == nil {
		return 0
	}

	// 检测循环引用
	if visiting[typ] {
		// 正在访问这个类型，说明遇到了递归
		// 对于递归类型，返回一个合理的默认值
		// 通常递归是通过指针实现的，所以返回指针大小
		return int64(types.PtrSize)
	}

	// 标记正在访问
	visiting[typ] = true
	defer func() { delete(visiting, typ) }()

	// 特殊处理：如果是占位符，返回指针大小
	// 因为递归引用通常是通过指针实现的
	if isForwardRefPlaceholder(typ) {
		return int64(types.PtrSize)
	}

	// 对于已经计算过大小的类型，直接返回
	if typ.Size() > 0 {
		return typ.Size()
	}

	// 根据类型种类计算大小
	switch typ.Kind() {
	case types.TPTR:
		// 指针类型：固定大小，不递归到元素类型
		// 这是递归类型的关键：指针打破了无限递归
		return int64(types.PtrSize)

	case types.TSLICE:
		// 切片：{ptr, len, cap}
		return int64(types.PtrSize) * 3

	case types.TSTRING:
		// 字符串：{ptr, len}
		return int64(types.PtrSize) * 2

	case types.TINTER:
		// 接口：{type, data}
		return int64(types.PtrSize) * 2

	case types.TMAP, types.TCHAN, types.TFUNC:
		// map, channel, func 都是指针
		return int64(types.PtrSize)

	case types.TARRAY:
		// 数组：元素大小 × 元素个数
		elemSize := safeCalcTypeSize(typ.Elem(), visiting)
		return elemSize * typ.NumElem()

	case types.TSTRUCT:
		// 结构体：递归计算所有字段
		totalSize := int64(0)
		for _, f := range typ.Fields() {
			fieldSize := safeCalcTypeSize(f.Type, visiting)
			// 简化：不考虑精确对齐和填充，只是累加
			totalSize += fieldSize
		}
		// 向上对齐到指针大小
		if totalSize > 0 && totalSize%int64(types.PtrSize) != 0 {
			totalSize = ((totalSize / int64(types.PtrSize)) + 1) * int64(types.PtrSize)
		}
		return totalSize

	case types.TINT, types.TUINT:
		return int64(types.PtrSize) // 平台相关

	case types.TINT8, types.TUINT8, types.TBOOL:
		return 1

	case types.TINT16, types.TUINT16:
		return 2

	case types.TINT32, types.TUINT32, types.TFLOAT32:
		return 4

	case types.TINT64, types.TUINT64, types.TFLOAT64, types.TCOMPLEX64:
		return 8

	case types.TCOMPLEX128:
		return 16

	case types.TUINTPTR, types.TUNSAFEPTR:
		return int64(types.PtrSize)

	default:
		// 对于其他类型，尝试调用 CalcSize
		// 如果失败，返回指针大小作为安全默认值
		defer func() { _ = recover() }()

		types.CalcSize(typ)
		sz := typ.Size()
		if sz > 0 {
			return sz
		}
		return int64(types.PtrSize) // 安全默认值
	}
}

// hasAnyPointers 修复版：使用真实的 Type 系统判断
func hasAnyPointers(variants []*syntax.EnumVariant) bool {
	for _, v := range variants {
		if v.Type == nil {
			continue
		}
		typ := convertSyntaxTypeToIRType(v.Type)
		if typ == nil {
			continue
		}
		if typ.HasPointers() {
			return true
		}
	}
	return false
}

// convertSyntaxTypeToIRType 增强版
func convertSyntaxTypeToIRType(expr syntax.Expr) *types.Type {
	if expr == nil {
		return nil
	}

	switch t := expr.(type) {
	case *syntax.Name:
		switch t.Value {
		case "bool":
			return types.Types[types.TBOOL]
		case "int":
			return types.Types[types.TINT]
		case "int8":
			return types.Types[types.TINT8]
		case "int16":
			return types.Types[types.TINT16]
		case "int32", "rune":
			return types.Types[types.TINT32]
		case "int64":
			return types.Types[types.TINT64]
		case "uint":
			return types.Types[types.TUINT]
		case "uint8", "byte":
			return types.Types[types.TUINT8]
		case "uint16":
			return types.Types[types.TUINT16]
		case "uint32":
			return types.Types[types.TUINT32]
		case "uint64":
			return types.Types[types.TUINT64]
		case "uintptr":
			return types.Types[types.TUINTPTR]
		case "float32":
			return types.Types[types.TFLOAT32]
		case "float64":
			return types.Types[types.TFLOAT64]
		case "complex64":
			return types.Types[types.TCOMPLEX64]
		case "complex128":
			return types.Types[types.TCOMPLEX128]
		case "string":
			return types.Types[types.TSTRING]
		case "any":
			return types.Types[types.TINTER] // alias for interface{}
		default:
			sym := typecheck.Lookup(t.Value)
			if sym.Def != nil {
				if n, ok := sym.Def.(*ir.Name); ok {
					return n.Type()
				}
			}

			// 检查缓存，避免重复创建
			typeName := t.Value
			if cached, ok := forwardRefCache[typeName]; ok {
				return cached
			}

			// 创建占位类型（TFORW）。types.NewNamed 需要一个 types.Object（通常是 *ir.Name）
			// 这里用一个 OTYPE 的 ir.Name 做占位。
			typeObj := ir.NewDeclNameAt(src.NoXPos, ir.OTYPE, sym)
			placeholderType := types.NewNamed(typeObj)

			// 设置临时底层类型
			// 注意：这个类型会在后续的类型检查阶段被正确的类型替换
			// 使用 TUINT8 作为占位，因为它最小且不包含指针
			placeholderType.SetUnderlying(types.Types[types.TUINT8])

			// 缓存占位类型
			forwardRefCache[typeName] = placeholderType

			// 发出信息性警告（不是错误）
			// 这在递归类型定义中是完全合法的
			if base.Flag.LowerM != 0 { // 仅在 verbose 模式下输出
				base.WarnfAt(src.NoXPos, "forward reference to type %s, created placeholder", typeName)
			}

			return placeholderType
		}

	case *syntax.SelectorExpr:
		// 这里无法在 noder 早期阶段可靠解析 imported pkg.Type（需要 types2.Info/导入器）
		// 先返回 nil，让上层决定报错/降级处理。
		return nil

	case *syntax.IndexExpr:
		// 泛型实例化支持: NDArray[int]
		baseType := convertSyntaxTypeToIRType(t.X)
		if baseType == nil {
			return nil
		}

		var typeArgs []*types.Type
		if list, ok := t.Index.(*syntax.ListExpr); ok {
			for _, arg := range list.ElemList {
				typeArgs = append(typeArgs, convertSyntaxTypeToIRType(arg))
			}
		} else {
			typeArgs = append(typeArgs, convertSyntaxTypeToIRType(t.Index))
		}

		// cmd/compile/internal/types 这里没有公开的“实例化 types.Type”构造器。
		// 在 noder 早期阶段先降级为 baseType，让后续类型系统/检查器决定如何处理。
		_ = typeArgs
		return baseType

	case *syntax.ArrayType:
		elem := convertSyntaxTypeToIRType(t.Elem)
		if t.Len == nil {
			return types.NewSlice(elem)
		}
		// 数组长度解析
		if lit, ok := t.Len.(*syntax.BasicLit); ok {
			n, _ := strconv.ParseInt(lit.Value, 0, 64)
			return types.NewArray(elem, n)
		}
		base.ErrorfAt(src.NoXPos, 0, "complex array length not supported in enum")
		return types.NewSlice(elem)

	case *syntax.StructType:
		// 简化的结构体处理
		var fields []*types.Field
		for _, f := range t.FieldList {
			ft := convertSyntaxTypeToIRType(f.Type)
			var sym *types.Sym
			if f.Name != nil {
				sym = typecheck.Lookup(f.Name.Value)
			}
			fields = append(fields, types.NewField(src.NoXPos, sym, ft))
		}
		return types.NewStruct(fields)

	case *syntax.InterfaceType:
		return types.NewInterface(nil)

	case *syntax.Operation:
		if t.Op == syntax.Mul && t.Y == nil {
			return types.NewPtr(convertSyntaxTypeToIRType(t.X))
		}
	}

	return nil
}
