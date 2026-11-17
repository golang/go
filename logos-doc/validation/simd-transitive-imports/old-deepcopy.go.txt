// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package midway

import (
	"fmt"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// DeepCopier clones syntax nodes and maintains types2.Info mappings.
type DeepCopier struct {
	VecLen   int
	info     *types2.Info
	pkg      *types2.Package
	analyzer *Analyzer
	suffix   string

	vars map[*types2.Var]*types2.Var
}

func NewDeepCopier(pkg *types2.Package, info *types2.Info, vecLen int, analyzer *Analyzer, suffix string) *DeepCopier {
	return &DeepCopier{
		VecLen:   vecLen,
		info:     info,
		pkg:      pkg,
		analyzer: analyzer,
		suffix:   suffix,
		vars:     make(map[*types2.Var]*types2.Var),
	}
}

func (c *DeepCopier) registerDef(newName *syntax.Name, oldName *syntax.Name) {
	if oldName == nil || newName == nil {
		return
	}
	if oldObj := c.info.Defs[oldName]; oldObj != nil {
		if val, isVar := oldObj.(*types2.Var); isVar {
			newObj := types2.NewVar(newName.Pos(), c.pkg, newName.Value, val.Type())
			c.vars[val] = newObj
			c.info.Defs[newName] = newObj
		} else {
			c.info.Defs[newName] = oldObj
		}
	}
}

func (c *DeepCopier) mapUse(newName *syntax.Name, oldName *syntax.Name) {
	if oldName == nil || newName == nil {
		return
	}
	if oldObj := c.info.Uses[oldName]; oldObj != nil {
		if val, isVar := oldObj.(*types2.Var); isVar && c.vars[val] != nil {
			c.info.Uses[newName] = c.vars[val]
		} else {
			c.info.Uses[newName] = oldObj
		}
	}
}

// OnName rewrites "dependent" and SIMD names to their architecture-specific version.
func (c *DeepCopier) OnName(id *syntax.Name) *syntax.Name {
	obj := c.info.Uses[id]
	if obj == nil {
		obj = c.info.Defs[id]
	}
	if obj == nil {
		return nil
	}
	// Don't rename methods of dependent types
	if c.analyzer.isDependentMethod[obj] {
		return nil
	}

	if c.analyzer.isDependentObj[obj] || isBaseSimdTypeObj(obj) {
		newId := syntax.NewName(id.Pos(), id.Value+c.suffix)
		// Object link will be handled manually in deepcopier Use/Def mapper
		if base.Debug.Simd > 0 {
			base.Warn("%s: rewriting name %s to %s", id.Pos().String(), id.Value, newId.Value)
		}
		return newId
	}
	return nil
}

// OnNameExpr rewrites references to simd.<simd type> into
// <bridge package>.<size-dependent-type>.
func (c *DeepCopier) OnNameExpr(id *syntax.Name) syntax.Expr {
	obj := c.info.Uses[id]
	if obj == nil {
		obj = c.info.Defs[id]
	}
	if obj == nil {
		return nil
	}

	if isBaseSimdTypeObj(obj) {
		// if it is a name, that means that this is in the simd package,
		// and the name must be replaced with a selector referencing
		// the architecture-dependent packages.
		name := id.Value
		width := nameToElemBitWidth(name)
		if width > 0 {
			archsimdId := syntax.NewName(id.Pos(), archPkg)
			if c.VecLen == 0 {
				// special case for emulation
				newSel := &syntax.SelectorExpr{
					X:   archsimdId,
					Sel: id, // name is unchanged for emulation
				}
				newSel.SetPos(id.Pos())
				return newSel
			}

			count := c.VecLen / width
			base := name[:len(name)-1]
			newName := fmt.Sprintf("%sx%d", base, count)
			newSelId := syntax.NewName(id.Pos(), newName)
			newSel := &syntax.SelectorExpr{
				X:   archsimdId,
				Sel: newSelId,
			}
			newSel.SetPos(id.Pos())
			return newSel
		}
	}

	if c.analyzer.isDependentObj[obj] {
		newId := syntax.NewName(id.Pos(), id.Value+c.suffix)
		// Object link will be handled manually in deepcopier Use/Def mapper
		if base.Debug.Simd > 0 {
			base.Warn("%s: rewriting name %s to %s", id.Pos().String(), id.Value, newId.Value)
		}
		return newId
	}
	return nil
}

// OnSelector is looking for simd.Something, to be rewritten into
// appropriately.  Note that this will not work properly within the simd
// package because there is no "simd." selection there.
func (c *DeepCopier) OnSelector(se *syntax.SelectorExpr) syntax.Expr {
	if x, ok := se.X.(*syntax.Name); ok {
		obj := c.info.Uses[x]
		if pkgName, isPkg := obj.(*types2.PkgName); isPkg && pkgName.Imported().Path() == simdPkg {
			// This first little bit detects name = Load-Type-Width-s-{,Part}
			// and converts the name to Type-Width-s (for nameToWidth), sets isLoad,
			// and initializes the suffix appropriately.
			prefix := ""
			nameSuffix := ""
			name := se.Sel.Value
			end := len(name)
			if strings.HasPrefix(name, "Load") {
				prefix = "Load"
				if strings.HasSuffix(name, "Part") {
					end = strings.Index(name, "Part")
					nameSuffix = "Part"
				}
				name = name[len("Load"):end]
			}
			if strings.HasPrefix(name, "Broadcast") {
				prefix = "Broadcast"
				name = name[len("Broadcast"):end]
			}

			width := nameToElemBitWidth(name)
			if width > 0 {
				archsimdId := syntax.NewName(se.Pos(), archPkg)
				if c.VecLen == 0 {
					// emulated instead, name is unchanged
					newSel := &syntax.SelectorExpr{
						X:   archsimdId,
						Sel: se.Sel,
					}
					newSel.SetPos(se.Pos())
					return newSel
				}

				count := c.VecLen / width
				base := name[:len(name)-1]
				newName := fmt.Sprintf("%sx%d", base, count)
				newName = prefix + newName + nameSuffix

				newSelId := syntax.NewName(se.Sel.Pos(), newName)

				newSel := &syntax.SelectorExpr{
					X:   archsimdId,
					Sel: newSelId,
				}
				newSel.SetPos(se.Pos())
				return newSel
			}
		}
	}
	return nil
}

func (c *DeepCopier) CopyDecl(d syntax.Decl) syntax.Decl {
	if d == nil {
		return nil
	}
	switch d := d.(type) {
	case *syntax.FuncDecl:
		return c.CopyFuncDecl(d)
	case *syntax.VarDecl:
		return c.CopyVarDecl(d)
	case *syntax.TypeDecl:
		return c.CopyTypeDecl(d)
	case *syntax.ConstDecl:
		return c.CopyConstDecl(d)
	case *syntax.ImportDecl:
		newD := &syntax.ImportDecl{
			Group:        d.Group,
			Pragma:       d.Pragma,
			LocalPkgName: c.CopyName(d.LocalPkgName, false),
			Path:         c.CopyExpr(d.Path).(*syntax.BasicLit),
		}
		newD.SetPos(d.Pos())
		return newD
	default:
		return d
	}
}

func (c *DeepCopier) CopyVarDecl(d *syntax.VarDecl) *syntax.VarDecl {
	newD := &syntax.VarDecl{
		Group:  d.Group,
		Pragma: d.Pragma,
		Type:   c.CopyExpr(d.Type),
		Values: c.CopyExpr(d.Values),
	}
	newD.SetPos(d.Pos())
	for _, n := range d.NameList {
		newN := c.CopyName(n, true)
		newD.NameList = append(newD.NameList, newN)
	}
	return newD
}

func (c *DeepCopier) CopyTypeDecl(d *syntax.TypeDecl) *syntax.TypeDecl {
	newD := &syntax.TypeDecl{
		Group:      d.Group,
		Pragma:     d.Pragma,
		Name:       c.CopyName(d.Name, true),
		TParamList: c.CopyFieldList(d.TParamList),
		Alias:      d.Alias,
		Type:       c.CopyExpr(d.Type),
	}
	newD.SetPos(d.Pos())
	return newD
}

func (c *DeepCopier) CopyConstDecl(d *syntax.ConstDecl) *syntax.ConstDecl {
	newD := &syntax.ConstDecl{
		Group:  d.Group,
		Pragma: d.Pragma,
		Type:   c.CopyExpr(d.Type),
		Values: c.CopyExpr(d.Values),
	}
	newD.SetPos(d.Pos())
	for _, n := range d.NameList {
		newD.NameList = append(newD.NameList, c.CopyName(n, true))
	}
	return newD
}

func (c *DeepCopier) CopyFuncDecl(d *syntax.FuncDecl) *syntax.FuncDecl {
	newD := &syntax.FuncDecl{
		Pragma:     d.Pragma,
		Recv:       c.CopyField(d.Recv),
		Name:       c.CopyName(d.Name, true),
		TParamList: c.CopyFieldList(d.TParamList),
		Type:       c.CopyExpr(d.Type).(*syntax.FuncType),
	}
	newD.SetPos(d.Pos())

	// Create and register new types2.Func
	if oldFuncObj, ok := c.info.Defs[d.Name].(*types2.Func); ok {
		newFuncObj := types2.NewFunc(newD.Name.Pos(), c.pkg, newD.Name.Value, oldFuncObj.Type().(*types2.Signature))
		c.info.Defs[newD.Name] = newFuncObj
	}

	newD.Body = c.CopyBlockStmt(d.Body)
	return newD
}

func (c *DeepCopier) CopyName(id *syntax.Name, isDef bool) *syntax.Name {
	if id == nil {
		return nil
	}
	if match := c.OnName(id); match != nil {
		match.SetPos(id.Pos())
		if isDef {
			c.registerDef(match, id)
		} else {
			c.mapUse(match, id)
		}
		return match
	}
	newId := syntax.NewName(id.Pos(), id.Value)
	if isDef {
		c.registerDef(newId, id)
	} else {
		c.mapUse(newId, id)
	}
	return newId
}

func (c *DeepCopier) CopyNameExpr(id *syntax.Name) syntax.Expr {
	if !c.analyzer.inSimd {
		return c.CopyName(id, false)
	}
	if id == nil {
		return nil
	}

	if match := c.OnNameExpr(id); match != nil {
		match.SetPos(id.Pos())
		if n, ok := match.(*syntax.Name); ok {
			c.mapUse(n, id)
		}
		return match
	}

	newId := syntax.NewName(id.Pos(), id.Value)
	c.mapUse(newId, id)
	return newId
}

func (c *DeepCopier) CopyExpr(e syntax.Expr) syntax.Expr {
	if e == nil {
		return nil
	}
	var newE syntax.Expr
	switch e := e.(type) {
	case *syntax.Name:
		return c.CopyNameExpr(e)
	case *syntax.BasicLit:
		newLit := &syntax.BasicLit{Value: e.Value, Kind: e.Kind, Bad: e.Bad}
		newE = newLit
	case *syntax.CompositeLit:
		newLit := &syntax.CompositeLit{
			Type:   c.CopyExpr(e.Type),
			NKeys:  e.NKeys,
			Rbrace: e.Rbrace,
		}
		for _, el := range e.ElemList {
			newLit.ElemList = append(newLit.ElemList, c.CopyExpr(el))
		}
		newE = newLit
	case *syntax.KeyValueExpr:
		newE = &syntax.KeyValueExpr{Key: c.CopyExpr(e.Key), Value: c.CopyExpr(e.Value)}
	case *syntax.FuncLit:
		newE = &syntax.FuncLit{Type: c.CopyExpr(e.Type).(*syntax.FuncType), Body: c.CopyBlockStmt(e.Body)}
	case *syntax.ParenExpr:
		newE = &syntax.ParenExpr{X: c.CopyExpr(e.X)}
	case *syntax.SelectorExpr:
		if sub := c.OnSelector(e); sub != nil {
			sub.SetPos(e.Pos())
			if sel := c.info.Selections[e]; sel != nil {
				c.info.Selections[sub.(*syntax.SelectorExpr)] = sel
			}
			return sub
		}
		newSel := &syntax.SelectorExpr{X: c.CopyExpr(e.X), Sel: c.CopyName(e.Sel, false)}
		if sel := c.info.Selections[e]; sel != nil {
			c.info.Selections[newSel] = sel
		}
		newE = newSel
	case *syntax.IndexExpr:
		newE = &syntax.IndexExpr{X: c.CopyExpr(e.X), Index: c.CopyExpr(e.Index)}
	case *syntax.SliceExpr:
		newE = &syntax.SliceExpr{
			X:     c.CopyExpr(e.X),
			Index: [3]syntax.Expr{c.CopyExpr(e.Index[0]), c.CopyExpr(e.Index[1]), c.CopyExpr(e.Index[2])},
			Full:  e.Full,
		}
	case *syntax.AssertExpr:
		newE = &syntax.AssertExpr{X: c.CopyExpr(e.X), Type: c.CopyExpr(e.Type)}
	case *syntax.TypeSwitchGuard:
		newE = &syntax.TypeSwitchGuard{Lhs: c.CopyName(e.Lhs, true), X: c.CopyExpr(e.X)}
	case *syntax.Operation:
		newE = &syntax.Operation{Op: e.Op, X: c.CopyExpr(e.X), Y: c.CopyExpr(e.Y)}
	case *syntax.CallExpr:
		newCall := &syntax.CallExpr{
			Fun:     c.CopyExpr(e.Fun),
			HasDots: e.HasDots,
		}
		for _, a := range e.ArgList {
			newCall.ArgList = append(newCall.ArgList, c.CopyExpr(a))
		}
		newE = newCall
	case *syntax.ListExpr:
		newList := &syntax.ListExpr{}
		for _, el := range e.ElemList {
			newList.ElemList = append(newList.ElemList, c.CopyExpr(el))
		}
		newE = newList
	case *syntax.ArrayType:
		newE = &syntax.ArrayType{Len: c.CopyExpr(e.Len), Elem: c.CopyExpr(e.Elem)}
	case *syntax.SliceType:
		newE = &syntax.SliceType{Elem: c.CopyExpr(e.Elem)}
	case *syntax.DotsType:
		newE = &syntax.DotsType{Elem: c.CopyExpr(e.Elem)}
	case *syntax.StructType:
		newE = &syntax.StructType{
			FieldList: c.CopyFieldList(e.FieldList),
			TagList:   e.TagList, // Shallow copy for tags is fine usually
		}
	case *syntax.InterfaceType:
		newE = &syntax.InterfaceType{MethodList: c.CopyFieldList(e.MethodList)}
	case *syntax.FuncType:
		newE = &syntax.FuncType{
			ParamList:  c.CopyFieldList(e.ParamList),
			ResultList: c.CopyFieldList(e.ResultList),
		}
	case *syntax.MapType:
		newE = &syntax.MapType{Key: c.CopyExpr(e.Key), Value: c.CopyExpr(e.Value)}
	case *syntax.ChanType:
		newE = &syntax.ChanType{Dir: e.Dir, Elem: c.CopyExpr(e.Elem)}
	case *syntax.BadExpr:
		newE = &syntax.BadExpr{}
	default:
		newE = e
	}
	newE.SetPos(e.Pos())
	return newE
}

func (c *DeepCopier) CopyStmt(s syntax.Stmt) syntax.Stmt {
	if s == nil {
		return nil
	}
	var newS syntax.Stmt
	switch s := s.(type) {
	case *syntax.DeclStmt:
		newDeclList := make([]syntax.Decl, len(s.DeclList))
		for i, v := range s.DeclList {
			newDeclList[i] = c.CopyDecl(v)
		}
		newS = &syntax.DeclStmt{DeclList: newDeclList}
	case *syntax.ExprStmt:
		newS = &syntax.ExprStmt{X: c.CopyExpr(s.X)}
	case *syntax.SendStmt:
		newS = &syntax.SendStmt{Chan: c.CopyExpr(s.Chan), Value: c.CopyExpr(s.Value)}
	case *syntax.AssignStmt:
		newS = &syntax.AssignStmt{Op: s.Op, Lhs: c.CopyExpr(s.Lhs), Rhs: c.CopyExpr(s.Rhs)}
	case *syntax.ReturnStmt:
		newS = &syntax.ReturnStmt{Results: c.CopyExpr(s.Results)}
	case *syntax.BranchStmt:
		// TODO this is broken
		newS = &syntax.BranchStmt{Tok: s.Tok, Label: c.CopyName(s.Label, false), Target: nil} // Targets need fix-up
	case *syntax.CallStmt:
		newS = &syntax.CallStmt{Tok: s.Tok, Call: c.CopyExpr(s.Call), DeferAt: c.CopyExpr(s.DeferAt)}
	case *syntax.IfStmt:
		newS = &syntax.IfStmt{
			Init: c.CopySimpleStmt(s.Init),
			Cond: c.CopyExpr(s.Cond),
			Then: c.CopyBlockStmt(s.Then),
			Else: c.CopyStmt(s.Else),
		}
	case *syntax.ForStmt:
		newS = &syntax.ForStmt{
			Init: c.CopySimpleStmt(s.Init),
			Cond: c.CopyExpr(s.Cond),
			Post: c.CopySimpleStmt(s.Post),
			Body: c.CopyBlockStmt(s.Body),
		}
	case *syntax.SwitchStmt:
		newS = &syntax.SwitchStmt{
			Init:   c.CopySimpleStmt(s.Init),
			Tag:    c.CopyExpr(s.Tag),
			Body:   c.CopyCaseClauses(s.Body),
			Rbrace: s.Rbrace,
		}
	case *syntax.SelectStmt:
		newS = &syntax.SelectStmt{
			Body:   c.CopyCommClauses(s.Body),
			Rbrace: s.Rbrace,
		}
	case *syntax.EmptyStmt:
		newS = &syntax.EmptyStmt{}
	case *syntax.LabeledStmt:
		newS = &syntax.LabeledStmt{Label: c.CopyName(s.Label, true), Stmt: c.CopyStmt(s.Stmt)} // Labels are defs
	case *syntax.BlockStmt:
		return c.CopyBlockStmt(s)
	default:
		newS = s
	}
	newS.SetPos(s.Pos())
	return newS
}

func (c *DeepCopier) CopySimpleStmt(s syntax.SimpleStmt) syntax.SimpleStmt {
	if s == nil {
		return nil
	}
	switch s := s.(type) {
	case *syntax.RangeClause:
		newS := &syntax.RangeClause{
			Def: s.Def,
			X:   c.CopyExpr(s.X),
		}
		// In a range clause, Lhs may contain definitions if Def is true.
		if list, ok := s.Lhs.(*syntax.ListExpr); ok && s.Def {
			newList := &syntax.ListExpr{}
			for _, el := range list.ElemList {
				if id, ok := el.(*syntax.Name); ok {
					newList.ElemList = append(newList.ElemList, c.CopyName(id, true))
				} else {
					newList.ElemList = append(newList.ElemList, c.CopyExpr(el))
				}
			}
			newS.Lhs = newList
		} else if id, ok := s.Lhs.(*syntax.Name); ok && s.Def {
			newS.Lhs = c.CopyName(id, true)
		} else {
			newS.Lhs = c.CopyExpr(s.Lhs)
		}
		newS.Lhs.SetPos(s.Lhs.Pos())
		newS.SetPos(s.Pos())
		return newS
	case *syntax.AssignStmt:
		// Check for :=
		isDef := false
		if list, ok := s.Lhs.(*syntax.ListExpr); ok {
			for _, el := range list.ElemList {
				if id, ok := el.(*syntax.Name); ok && c.info.Defs[id] != nil {
					isDef = true
					break
				}
			}
		} else if id, ok := s.Lhs.(*syntax.Name); ok && c.info.Defs[id] != nil {
			isDef = true
		}

		newS := &syntax.AssignStmt{Op: s.Op, Rhs: c.CopyExpr(s.Rhs)}
		if isDef {
			if list, ok := s.Lhs.(*syntax.ListExpr); ok {
				newList := &syntax.ListExpr{}
				for _, el := range list.ElemList {
					if id, ok := el.(*syntax.Name); ok && c.info.Defs[id] != nil {
						newList.ElemList = append(newList.ElemList, c.CopyName(id, true))
					} else {
						newList.ElemList = append(newList.ElemList, c.CopyExpr(el))
					}
				}
				newS.Lhs = newList
			} else if id, ok := s.Lhs.(*syntax.Name); ok {
				newS.Lhs = c.CopyName(id, true)
			}
		} else {
			newS.Lhs = c.CopyExpr(s.Lhs)
		}
		newS.Lhs.SetPos(s.Lhs.Pos())
		newS.SetPos(s.Pos())
		return newS
	default:
		return c.CopyStmt(s).(syntax.SimpleStmt)
	}
}

func (c *DeepCopier) CopyCaseClauses(list []*syntax.CaseClause) []*syntax.CaseClause {
	var newList []*syntax.CaseClause
	for _, cc := range list {
		newC := &syntax.CaseClause{Cases: c.CopyExpr(cc.Cases), Colon: cc.Colon}
		for _, b := range cc.Body {
			newC.Body = append(newC.Body, c.CopyStmt(b))
		}
		newC.SetPos(cc.Pos())
		newList = append(newList, newC)
	}
	return newList
}

func (c *DeepCopier) CopyCommClauses(list []*syntax.CommClause) []*syntax.CommClause {
	var newList []*syntax.CommClause
	for _, cc := range list {
		newC := &syntax.CommClause{Comm: c.CopySimpleStmt(cc.Comm), Colon: cc.Colon}
		for _, b := range cc.Body {
			newC.Body = append(newC.Body, c.CopyStmt(b))
		}
		newC.SetPos(cc.Pos())
		newList = append(newList, newC)
	}
	return newList
}

func (c *DeepCopier) CopyBlockStmt(b *syntax.BlockStmt) *syntax.BlockStmt {
	if b == nil {
		return nil
	}
	newB := &syntax.BlockStmt{Rbrace: b.Rbrace}
	for _, s := range b.List {
		newB.List = append(newB.List, c.CopyStmt(s))
	}
	newB.SetPos(b.Pos())
	return newB
}

func (c *DeepCopier) CopyFieldList(f []*syntax.Field) []*syntax.Field {
	if f == nil {
		return nil
	}
	var newF []*syntax.Field
	for _, field := range f {
		newF = append(newF, c.CopyField(field))
	}
	return newF
}

func (c *DeepCopier) CopyField(f *syntax.Field) *syntax.Field {
	if f == nil {
		return nil
	}
	newF := &syntax.Field{
		Name: c.CopyName(f.Name, true),
		Type: c.CopyExpr(f.Type),
	}
	newF.SetPos(f.Pos())
	return newF
}
