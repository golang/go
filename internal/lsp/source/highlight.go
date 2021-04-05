// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/protocol"
	errors "golang.org/x/xerrors"
)

func Highlight(ctx context.Context, snapshot Snapshot, fh FileHandle, pos protocol.Position) ([]protocol.Range, error) {
	ctx, done := event.Start(ctx, "source.Highlight")
	defer done()

	// Don't use GetParsedFile because it uses TypecheckWorkspace, and we
	// always want fully parsed files for highlight, regardless of whether
	// the file belongs to a workspace package.
	pkg, err := snapshot.PackageForFile(ctx, fh.URI(), TypecheckFull, WidestPackage)
	if err != nil {
		return nil, errors.Errorf("getting package for Highlight: %w", err)
	}
	pgf, err := pkg.File(fh.URI())
	if err != nil {
		return nil, errors.Errorf("getting file for Highlight: %w", err)
	}

	spn, err := pgf.Mapper.PointSpan(pos)
	if err != nil {
		return nil, err
	}
	rng, err := spn.Range(pgf.Mapper.Converter)
	if err != nil {
		return nil, err
	}
	path, _ := astutil.PathEnclosingInterval(pgf.File, rng.Start, rng.Start)
	if len(path) == 0 {
		return nil, fmt.Errorf("no enclosing position found for %v:%v", int(pos.Line), int(pos.Character))
	}
	// If start == end for astutil.PathEnclosingInterval, the 1-char interval
	// following start is used instead. As a result, we might not get an exact
	// match so we should check the 1-char interval to the left of the passed
	// in position to see if that is an exact match.
	if _, ok := path[0].(*ast.Ident); !ok {
		if p, _ := astutil.PathEnclosingInterval(pgf.File, rng.Start-1, rng.Start-1); p != nil {
			switch p[0].(type) {
			case *ast.Ident, *ast.SelectorExpr:
				path = p // use preceding ident/selector
			}
		}
	}
	result, err := highlightPath(pkg, path)
	if err != nil {
		return nil, err
	}
	var ranges []protocol.Range
	for rng := range result {
		mRng, err := posToMappedRange(snapshot, pkg, rng.start, rng.end)
		if err != nil {
			return nil, err
		}
		pRng, err := mRng.Range()
		if err != nil {
			return nil, err
		}
		ranges = append(ranges, pRng)
	}
	return ranges, nil
}

func highlightPath(pkg Package, path []ast.Node) (map[posRange]struct{}, error) {
	result := make(map[posRange]struct{})
	switch node := path[0].(type) {
	case *ast.BasicLit:
		if len(path) > 1 {
			if _, ok := path[1].(*ast.ImportSpec); ok {
				err := highlightImportUses(pkg, path, result)
				return result, err
			}
		}
		highlightFuncControlFlow(path, result)
	case *ast.ReturnStmt, *ast.FuncDecl, *ast.FuncType:
		highlightFuncControlFlow(path, result)
	case *ast.Ident:
		highlightIdentifiers(pkg, path, result)
	case *ast.ForStmt, *ast.RangeStmt:
		highlightLoopControlFlow(path, result)
	case *ast.SwitchStmt:
		highlightSwitchFlow(path, result)
	case *ast.BranchStmt:
		// BREAK can exit a loop, switch or select, while CONTINUE exit a loop so
		// these need to be handled separately. They can also be embedded in any
		// other loop/switch/select if they have a label. TODO: add support for
		// GOTO and FALLTHROUGH as well.
		if node.Label != nil {
			highlightLabeledFlow(node, result)
		} else {
			switch node.Tok {
			case token.BREAK:
				highlightUnlabeledBreakFlow(path, result)
			case token.CONTINUE:
				highlightLoopControlFlow(path, result)
			}
		}
	default:
		// If the cursor is in an unidentified area, return empty results.
		return nil, nil
	}
	return result, nil
}

type posRange struct {
	start, end token.Pos
}

func highlightFuncControlFlow(path []ast.Node, result map[posRange]struct{}) {
	var enclosingFunc ast.Node
	var returnStmt *ast.ReturnStmt
	var resultsList *ast.FieldList
	inReturnList := false

Outer:
	// Reverse walk the path till we get to the func block.
	for i, n := range path {
		switch node := n.(type) {
		case *ast.KeyValueExpr:
			// If cursor is in a key: value expr, we don't want control flow highlighting
			return
		case *ast.CallExpr:
			// If cusor is an arg in a callExpr, we don't want control flow highlighting.
			if i > 0 {
				for _, arg := range node.Args {
					if arg == path[i-1] {
						return
					}
				}
			}
		case *ast.Field:
			inReturnList = true
		case *ast.FuncLit:
			enclosingFunc = n
			resultsList = node.Type.Results
			break Outer
		case *ast.FuncDecl:
			enclosingFunc = n
			resultsList = node.Type.Results
			break Outer
		case *ast.ReturnStmt:
			returnStmt = node
			// If the cursor is not directly in a *ast.ReturnStmt, then
			// we need to know if it is within one of the values that is being returned.
			inReturnList = inReturnList || path[0] != returnStmt
		}
	}
	// Cursor is not in a function.
	if enclosingFunc == nil {
		return
	}
	// If the cursor is on a "return" or "func" keyword, we should highlight all of the exit
	// points of the function, including the "return" and "func" keywords.
	highlightAllReturnsAndFunc := path[0] == returnStmt || path[0] == enclosingFunc
	switch path[0].(type) {
	case *ast.Ident, *ast.BasicLit:
		// Cursor is in an identifier and not in a return statement or in the results list.
		if returnStmt == nil && !inReturnList {
			return
		}
	case *ast.FuncType:
		highlightAllReturnsAndFunc = true
	}
	// The user's cursor may be within the return statement of a function,
	// or within the result section of a function's signature.
	// index := -1
	var nodes []ast.Node
	if returnStmt != nil {
		for _, n := range returnStmt.Results {
			nodes = append(nodes, n)
		}
	} else if resultsList != nil {
		for _, n := range resultsList.List {
			nodes = append(nodes, n)
		}
	}
	_, index := nodeAtPos(nodes, path[0].Pos())

	// Highlight the correct argument in the function declaration return types.
	if resultsList != nil && -1 < index && index < len(resultsList.List) {
		rng := posRange{
			start: resultsList.List[index].Pos(),
			end:   resultsList.List[index].End(),
		}
		result[rng] = struct{}{}
	}
	// Add the "func" part of the func declaration.
	if highlightAllReturnsAndFunc {
		r := posRange{
			start: enclosingFunc.Pos(),
			end:   enclosingFunc.Pos() + token.Pos(len("func")),
		}
		result[r] = struct{}{}
	}
	ast.Inspect(enclosingFunc, func(n ast.Node) bool {
		// Don't traverse any other functions.
		switch n.(type) {
		case *ast.FuncDecl, *ast.FuncLit:
			return enclosingFunc == n
		}
		ret, ok := n.(*ast.ReturnStmt)
		if !ok {
			return true
		}
		var toAdd ast.Node
		// Add the entire return statement, applies when highlight the word "return" or "func".
		if highlightAllReturnsAndFunc {
			toAdd = n
		}
		// Add the relevant field within the entire return statement.
		if -1 < index && index < len(ret.Results) {
			toAdd = ret.Results[index]
		}
		if toAdd != nil {
			result[posRange{start: toAdd.Pos(), end: toAdd.End()}] = struct{}{}
		}
		return false
	})
}

func highlightUnlabeledBreakFlow(path []ast.Node, result map[posRange]struct{}) {
	// Reverse walk the path until we find closest loop, select, or switch.
	for _, n := range path {
		switch n.(type) {
		case *ast.ForStmt, *ast.RangeStmt:
			highlightLoopControlFlow(path, result)
			return // only highlight the innermost statement
		case *ast.SwitchStmt:
			highlightSwitchFlow(path, result)
			return
		case *ast.SelectStmt:
			// TODO: add highlight when breaking a select.
			return
		}
	}
}

func highlightLabeledFlow(node *ast.BranchStmt, result map[posRange]struct{}) {
	obj := node.Label.Obj
	if obj == nil || obj.Decl == nil {
		return
	}
	label, ok := obj.Decl.(*ast.LabeledStmt)
	if !ok {
		return
	}
	switch label.Stmt.(type) {
	case *ast.ForStmt, *ast.RangeStmt:
		highlightLoopControlFlow([]ast.Node{label.Stmt, label}, result)
	case *ast.SwitchStmt:
		highlightSwitchFlow([]ast.Node{label.Stmt, label}, result)
	}
}

func labelFor(path []ast.Node) *ast.Ident {
	if len(path) > 1 {
		if n, ok := path[1].(*ast.LabeledStmt); ok {
			return n.Label
		}
	}
	return nil
}

func highlightLoopControlFlow(path []ast.Node, result map[posRange]struct{}) {
	var loop ast.Node
	var loopLabel *ast.Ident
	stmtLabel := labelFor(path)
Outer:
	// Reverse walk the path till we get to the for loop.
	for i := range path {
		switch n := path[i].(type) {
		case *ast.ForStmt, *ast.RangeStmt:
			loopLabel = labelFor(path[i:])

			if stmtLabel == nil || loopLabel == stmtLabel {
				loop = n
				break Outer
			}
		}
	}
	if loop == nil {
		return
	}

	// Add the for statement.
	rng := posRange{
		start: loop.Pos(),
		end:   loop.Pos() + token.Pos(len("for")),
	}
	result[rng] = struct{}{}

	// Traverse AST to find branch statements within the same for-loop.
	ast.Inspect(loop, func(n ast.Node) bool {
		switch n.(type) {
		case *ast.ForStmt, *ast.RangeStmt:
			return loop == n
		case *ast.SwitchStmt, *ast.SelectStmt:
			return false
		}
		b, ok := n.(*ast.BranchStmt)
		if !ok {
			return true
		}
		if b.Label == nil || labelDecl(b.Label) == loopLabel {
			result[posRange{start: b.Pos(), end: b.End()}] = struct{}{}
		}
		return true
	})

	// Find continue statements in the same loop or switches/selects.
	ast.Inspect(loop, func(n ast.Node) bool {
		switch n.(type) {
		case *ast.ForStmt, *ast.RangeStmt:
			return loop == n
		}

		if n, ok := n.(*ast.BranchStmt); ok && n.Tok == token.CONTINUE {
			result[posRange{start: n.Pos(), end: n.End()}] = struct{}{}
		}
		return true
	})

	// We don't need to check other for loops if we aren't looking for labeled statements.
	if loopLabel == nil {
		return
	}

	// Find labeled branch statements in any loop.
	ast.Inspect(loop, func(n ast.Node) bool {
		b, ok := n.(*ast.BranchStmt)
		if !ok {
			return true
		}
		// statement with labels that matches the loop
		if b.Label != nil && labelDecl(b.Label) == loopLabel {
			result[posRange{start: b.Pos(), end: b.End()}] = struct{}{}
		}
		return true
	})
}

func highlightSwitchFlow(path []ast.Node, result map[posRange]struct{}) {
	var switchNode ast.Node
	var switchNodeLabel *ast.Ident
	stmtLabel := labelFor(path)
Outer:
	// Reverse walk the path till we get to the switch statement.
	for i := range path {
		switch n := path[i].(type) {
		case *ast.SwitchStmt:
			switchNodeLabel = labelFor(path[i:])
			if stmtLabel == nil || switchNodeLabel == stmtLabel {
				switchNode = n
				break Outer
			}
		}
	}
	// Cursor is not in a switch statement
	if switchNode == nil {
		return
	}

	// Add the switch statement.
	rng := posRange{
		start: switchNode.Pos(),
		end:   switchNode.Pos() + token.Pos(len("switch")),
	}
	result[rng] = struct{}{}

	// Traverse AST to find break statements within the same switch.
	ast.Inspect(switchNode, func(n ast.Node) bool {
		switch n.(type) {
		case *ast.SwitchStmt:
			return switchNode == n
		case *ast.ForStmt, *ast.RangeStmt, *ast.SelectStmt:
			return false
		}

		b, ok := n.(*ast.BranchStmt)
		if !ok || b.Tok != token.BREAK {
			return true
		}

		if b.Label == nil || labelDecl(b.Label) == switchNodeLabel {
			result[posRange{start: b.Pos(), end: b.End()}] = struct{}{}
		}
		return true
	})

	// We don't need to check other switches if we aren't looking for labeled statements.
	if switchNodeLabel == nil {
		return
	}

	// Find labeled break statements in any switch
	ast.Inspect(switchNode, func(n ast.Node) bool {
		b, ok := n.(*ast.BranchStmt)
		if !ok || b.Tok != token.BREAK {
			return true
		}

		if b.Label != nil && labelDecl(b.Label) == switchNodeLabel {
			result[posRange{start: b.Pos(), end: b.End()}] = struct{}{}
		}

		return true
	})
}

func labelDecl(n *ast.Ident) *ast.Ident {
	if n == nil {
		return nil
	}
	if n.Obj == nil {
		return nil
	}
	if n.Obj.Decl == nil {
		return nil
	}
	stmt, ok := n.Obj.Decl.(*ast.LabeledStmt)
	if !ok {
		return nil
	}
	return stmt.Label
}

func highlightImportUses(pkg Package, path []ast.Node, result map[posRange]struct{}) error {
	basicLit, ok := path[0].(*ast.BasicLit)
	if !ok {
		return errors.Errorf("highlightImportUses called with an ast.Node of type %T", basicLit)
	}
	ast.Inspect(path[len(path)-1], func(node ast.Node) bool {
		if imp, ok := node.(*ast.ImportSpec); ok && imp.Path == basicLit {
			result[posRange{start: node.Pos(), end: node.End()}] = struct{}{}
			return false
		}
		n, ok := node.(*ast.Ident)
		if !ok {
			return true
		}
		obj, ok := pkg.GetTypesInfo().ObjectOf(n).(*types.PkgName)
		if !ok {
			return true
		}
		if !strings.Contains(basicLit.Value, obj.Name()) {
			return true
		}
		result[posRange{start: n.Pos(), end: n.End()}] = struct{}{}
		return false
	})
	return nil
}

func highlightIdentifiers(pkg Package, path []ast.Node, result map[posRange]struct{}) error {
	id, ok := path[0].(*ast.Ident)
	if !ok {
		return errors.Errorf("highlightIdentifiers called with an ast.Node of type %T", id)
	}
	// Check if ident is inside return or func decl.
	highlightFuncControlFlow(path, result)

	// TODO: maybe check if ident is a reserved word, if true then don't continue and return results.

	idObj := pkg.GetTypesInfo().ObjectOf(id)
	pkgObj, isImported := idObj.(*types.PkgName)
	ast.Inspect(path[len(path)-1], func(node ast.Node) bool {
		if imp, ok := node.(*ast.ImportSpec); ok && isImported {
			highlightImport(pkgObj, imp, result)
		}
		n, ok := node.(*ast.Ident)
		if !ok {
			return true
		}
		if n.Name != id.Name {
			return false
		}
		if nObj := pkg.GetTypesInfo().ObjectOf(n); nObj == idObj {
			result[posRange{start: n.Pos(), end: n.End()}] = struct{}{}
		}
		return false
	})
	return nil
}

func highlightImport(obj *types.PkgName, imp *ast.ImportSpec, result map[posRange]struct{}) {
	if imp.Name != nil || imp.Path == nil {
		return
	}
	if !strings.Contains(imp.Path.Value, obj.Name()) {
		return
	}
	result[posRange{start: imp.Path.Pos(), end: imp.Path.End()}] = struct{}{}
}
