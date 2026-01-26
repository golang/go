// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package edge defines identifiers for each field of an ast.Node
// struct type that refers to another Node.
package edge

import (
	"fmt"
	"go/ast"
	"reflect"
)

// A Kind describes a field of an ast.Node struct.
type Kind uint8

// String returns a description of the edge kind.
func (k Kind) String() string {
	if k == Invalid {
		return "<invalid>"
	}
	info := fieldInfos[k]
	return fmt.Sprintf("%v.%s", info.nodeType.Elem().Name(), info.name)
}

// NodeType returns the pointer-to-struct type of the ast.Node implementation.
func (k Kind) NodeType() reflect.Type { return fieldInfos[k].nodeType }

// FieldName returns the name of the field.
func (k Kind) FieldName() string { return fieldInfos[k].name }

// FieldType returns the declared type of the field.
func (k Kind) FieldType() reflect.Type { return fieldInfos[k].fieldType }

// Get returns the direct child of n identified by (k, idx).
// n's type must match k.NodeType().
// idx must be a valid slice index, or -1 for a non-slice.
func (k Kind) Get(n ast.Node, idx int) ast.Node {
	if k.NodeType() != reflect.TypeOf(n) {
		panic(fmt.Sprintf("%v.Get(%T): invalid node type", k, n))
	}
	v := reflect.ValueOf(n).Elem().Field(fieldInfos[k].index)
	if idx != -1 {
		v = v.Index(idx) // asserts valid index
	} else {
		// (The type assertion below asserts that v is not a slice.)
	}
	return v.Interface().(ast.Node) // may be nil
}

const (
	Invalid Kind = iota // for nodes at the root of the traversal

	// Kinds are sorted alphabetically.
	// Numbering is not stable.
	// Each is named Type_Field, where Type is the
	// ast.Node struct type and Field is the name of the field

	ArrayType_Elt
	ArrayType_Len
	AssignStmt_Lhs
	AssignStmt_Rhs
	BinaryExpr_X
	BinaryExpr_Y
	BlockStmt_List
	BranchStmt_Label
	CallExpr_Args
	CallExpr_Fun
	CaseClause_Body
	CaseClause_List
	ChanType_Value
	CommClause_Body
	CommClause_Comm
	CommentGroup_List
	CompositeLit_Elts
	CompositeLit_Type
	DeclStmt_Decl
	DeferStmt_Call
	Ellipsis_Elt
	ExprStmt_X
	FieldList_List
	Field_Comment
	Field_Doc
	Field_Names
	Field_Tag
	Field_Type
	File_Decls
	File_Doc
	File_Name
	ForStmt_Body
	ForStmt_Cond
	ForStmt_Init
	ForStmt_Post
	FuncDecl_Body
	FuncDecl_Doc
	FuncDecl_Name
	FuncDecl_Recv
	FuncDecl_Type
	FuncLit_Body
	FuncLit_Type
	FuncType_Params
	FuncType_Results
	FuncType_TypeParams
	GenDecl_Doc
	GenDecl_Specs
	GoStmt_Call
	IfStmt_Body
	IfStmt_Cond
	IfStmt_Else
	IfStmt_Init
	ImportSpec_Comment
	ImportSpec_Doc
	ImportSpec_Name
	ImportSpec_Path
	IncDecStmt_X
	IndexExpr_Index
	IndexExpr_X
	IndexListExpr_Indices
	IndexListExpr_X
	InterfaceType_Methods
	KeyValueExpr_Key
	KeyValueExpr_Value
	LabeledStmt_Label
	LabeledStmt_Stmt
	MapType_Key
	MapType_Value
	ParenExpr_X
	RangeStmt_Body
	RangeStmt_Key
	RangeStmt_Value
	RangeStmt_X
	ReturnStmt_Results
	SelectStmt_Body
	SelectorExpr_Sel
	SelectorExpr_X
	SendStmt_Chan
	SendStmt_Value
	SliceExpr_High
	SliceExpr_Low
	SliceExpr_Max
	SliceExpr_X
	StarExpr_X
	StructType_Fields
	SwitchStmt_Body
	SwitchStmt_Init
	SwitchStmt_Tag
	TypeAssertExpr_Type
	TypeAssertExpr_X
	TypeSpec_Comment
	TypeSpec_Doc
	TypeSpec_Name
	TypeSpec_Type
	TypeSpec_TypeParams
	TypeSwitchStmt_Assign
	TypeSwitchStmt_Body
	TypeSwitchStmt_Init
	UnaryExpr_X
	ValueSpec_Comment
	ValueSpec_Doc
	ValueSpec_Names
	ValueSpec_Type
	ValueSpec_Values

	maxKind
)

// Assert that the encoding fits in 7 bits,
// as the inspector relies on this.
// (We are currently at 104.)
var _ = [1 << 7]struct{}{}[maxKind]

type fieldInfo struct {
	nodeType  reflect.Type // pointer-to-struct type of ast.Node implementation
	name      string
	index     int
	fieldType reflect.Type
}

func info[N ast.Node](fieldName string) fieldInfo {
	nodePtrType := reflect.TypeFor[N]()
	f, ok := nodePtrType.Elem().FieldByName(fieldName)
	if !ok {
		panic(fieldName)
	}
	return fieldInfo{nodePtrType, fieldName, f.Index[0], f.Type}
}

var fieldInfos = [...]fieldInfo{
	Invalid:               {},
	ArrayType_Elt:         info[*ast.ArrayType]("Elt"),
	ArrayType_Len:         info[*ast.ArrayType]("Len"),
	AssignStmt_Lhs:        info[*ast.AssignStmt]("Lhs"),
	AssignStmt_Rhs:        info[*ast.AssignStmt]("Rhs"),
	BinaryExpr_X:          info[*ast.BinaryExpr]("X"),
	BinaryExpr_Y:          info[*ast.BinaryExpr]("Y"),
	BlockStmt_List:        info[*ast.BlockStmt]("List"),
	BranchStmt_Label:      info[*ast.BranchStmt]("Label"),
	CallExpr_Args:         info[*ast.CallExpr]("Args"),
	CallExpr_Fun:          info[*ast.CallExpr]("Fun"),
	CaseClause_Body:       info[*ast.CaseClause]("Body"),
	CaseClause_List:       info[*ast.CaseClause]("List"),
	ChanType_Value:        info[*ast.ChanType]("Value"),
	CommClause_Body:       info[*ast.CommClause]("Body"),
	CommClause_Comm:       info[*ast.CommClause]("Comm"),
	CommentGroup_List:     info[*ast.CommentGroup]("List"),
	CompositeLit_Elts:     info[*ast.CompositeLit]("Elts"),
	CompositeLit_Type:     info[*ast.CompositeLit]("Type"),
	DeclStmt_Decl:         info[*ast.DeclStmt]("Decl"),
	DeferStmt_Call:        info[*ast.DeferStmt]("Call"),
	Ellipsis_Elt:          info[*ast.Ellipsis]("Elt"),
	ExprStmt_X:            info[*ast.ExprStmt]("X"),
	FieldList_List:        info[*ast.FieldList]("List"),
	Field_Comment:         info[*ast.Field]("Comment"),
	Field_Doc:             info[*ast.Field]("Doc"),
	Field_Names:           info[*ast.Field]("Names"),
	Field_Tag:             info[*ast.Field]("Tag"),
	Field_Type:            info[*ast.Field]("Type"),
	File_Decls:            info[*ast.File]("Decls"),
	File_Doc:              info[*ast.File]("Doc"),
	File_Name:             info[*ast.File]("Name"),
	ForStmt_Body:          info[*ast.ForStmt]("Body"),
	ForStmt_Cond:          info[*ast.ForStmt]("Cond"),
	ForStmt_Init:          info[*ast.ForStmt]("Init"),
	ForStmt_Post:          info[*ast.ForStmt]("Post"),
	FuncDecl_Body:         info[*ast.FuncDecl]("Body"),
	FuncDecl_Doc:          info[*ast.FuncDecl]("Doc"),
	FuncDecl_Name:         info[*ast.FuncDecl]("Name"),
	FuncDecl_Recv:         info[*ast.FuncDecl]("Recv"),
	FuncDecl_Type:         info[*ast.FuncDecl]("Type"),
	FuncLit_Body:          info[*ast.FuncLit]("Body"),
	FuncLit_Type:          info[*ast.FuncLit]("Type"),
	FuncType_Params:       info[*ast.FuncType]("Params"),
	FuncType_Results:      info[*ast.FuncType]("Results"),
	FuncType_TypeParams:   info[*ast.FuncType]("TypeParams"),
	GenDecl_Doc:           info[*ast.GenDecl]("Doc"),
	GenDecl_Specs:         info[*ast.GenDecl]("Specs"),
	GoStmt_Call:           info[*ast.GoStmt]("Call"),
	IfStmt_Body:           info[*ast.IfStmt]("Body"),
	IfStmt_Cond:           info[*ast.IfStmt]("Cond"),
	IfStmt_Else:           info[*ast.IfStmt]("Else"),
	IfStmt_Init:           info[*ast.IfStmt]("Init"),
	ImportSpec_Comment:    info[*ast.ImportSpec]("Comment"),
	ImportSpec_Doc:        info[*ast.ImportSpec]("Doc"),
	ImportSpec_Name:       info[*ast.ImportSpec]("Name"),
	ImportSpec_Path:       info[*ast.ImportSpec]("Path"),
	IncDecStmt_X:          info[*ast.IncDecStmt]("X"),
	IndexExpr_Index:       info[*ast.IndexExpr]("Index"),
	IndexExpr_X:           info[*ast.IndexExpr]("X"),
	IndexListExpr_Indices: info[*ast.IndexListExpr]("Indices"),
	IndexListExpr_X:       info[*ast.IndexListExpr]("X"),
	InterfaceType_Methods: info[*ast.InterfaceType]("Methods"),
	KeyValueExpr_Key:      info[*ast.KeyValueExpr]("Key"),
	KeyValueExpr_Value:    info[*ast.KeyValueExpr]("Value"),
	LabeledStmt_Label:     info[*ast.LabeledStmt]("Label"),
	LabeledStmt_Stmt:      info[*ast.LabeledStmt]("Stmt"),
	MapType_Key:           info[*ast.MapType]("Key"),
	MapType_Value:         info[*ast.MapType]("Value"),
	ParenExpr_X:           info[*ast.ParenExpr]("X"),
	RangeStmt_Body:        info[*ast.RangeStmt]("Body"),
	RangeStmt_Key:         info[*ast.RangeStmt]("Key"),
	RangeStmt_Value:       info[*ast.RangeStmt]("Value"),
	RangeStmt_X:           info[*ast.RangeStmt]("X"),
	ReturnStmt_Results:    info[*ast.ReturnStmt]("Results"),
	SelectStmt_Body:       info[*ast.SelectStmt]("Body"),
	SelectorExpr_Sel:      info[*ast.SelectorExpr]("Sel"),
	SelectorExpr_X:        info[*ast.SelectorExpr]("X"),
	SendStmt_Chan:         info[*ast.SendStmt]("Chan"),
	SendStmt_Value:        info[*ast.SendStmt]("Value"),
	SliceExpr_High:        info[*ast.SliceExpr]("High"),
	SliceExpr_Low:         info[*ast.SliceExpr]("Low"),
	SliceExpr_Max:         info[*ast.SliceExpr]("Max"),
	SliceExpr_X:           info[*ast.SliceExpr]("X"),
	StarExpr_X:            info[*ast.StarExpr]("X"),
	StructType_Fields:     info[*ast.StructType]("Fields"),
	SwitchStmt_Body:       info[*ast.SwitchStmt]("Body"),
	SwitchStmt_Init:       info[*ast.SwitchStmt]("Init"),
	SwitchStmt_Tag:        info[*ast.SwitchStmt]("Tag"),
	TypeAssertExpr_Type:   info[*ast.TypeAssertExpr]("Type"),
	TypeAssertExpr_X:      info[*ast.TypeAssertExpr]("X"),
	TypeSpec_Comment:      info[*ast.TypeSpec]("Comment"),
	TypeSpec_Doc:          info[*ast.TypeSpec]("Doc"),
	TypeSpec_Name:         info[*ast.TypeSpec]("Name"),
	TypeSpec_Type:         info[*ast.TypeSpec]("Type"),
	TypeSpec_TypeParams:   info[*ast.TypeSpec]("TypeParams"),
	TypeSwitchStmt_Assign: info[*ast.TypeSwitchStmt]("Assign"),
	TypeSwitchStmt_Body:   info[*ast.TypeSwitchStmt]("Body"),
	TypeSwitchStmt_Init:   info[*ast.TypeSwitchStmt]("Init"),
	UnaryExpr_X:           info[*ast.UnaryExpr]("X"),
	ValueSpec_Comment:     info[*ast.ValueSpec]("Comment"),
	ValueSpec_Doc:         info[*ast.ValueSpec]("Doc"),
	ValueSpec_Names:       info[*ast.ValueSpec]("Names"),
	ValueSpec_Type:        info[*ast.ValueSpec]("Type"),
	ValueSpec_Values:      info[*ast.ValueSpec]("Values"),
}
