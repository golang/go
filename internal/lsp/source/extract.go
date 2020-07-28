// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"go/types"
	"strings"
	"unicode"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/span"
)

func extractVariable(fset *token.FileSet, rng span.Range, src []byte, file *ast.File, pkg *types.Package, info *types.Info) (*analysis.SuggestedFix, error) {
	expr, path, ok, err := canExtractVariable(fset, rng, src, file, pkg, info)
	if !ok {
		return nil, fmt.Errorf("extractVariable: cannot extract %s: %v", fset.Position(rng.Start), err)
	}

	name := generateAvailableIdentifier(expr.Pos(), file, path, info, "x", 0)

	// Create new AST node for extracted code.
	var assignment string
	switch expr.(type) {
	case *ast.BasicLit, *ast.CompositeLit, *ast.IndexExpr,
		*ast.SliceExpr, *ast.UnaryExpr, *ast.BinaryExpr, *ast.SelectorExpr: // TODO: stricter rules for selectorExpr.
		assignStmt := &ast.AssignStmt{
			Lhs: []ast.Expr{ast.NewIdent(name)},
			Tok: token.DEFINE,
			Rhs: []ast.Expr{expr},
		}
		var buf bytes.Buffer
		if err := format.Node(&buf, fset, assignStmt); err != nil {
			return nil, err
		}
		assignment = buf.String()
	case *ast.CallExpr: // TODO: find number of return values and do according actions.
		return nil, nil
	default:
		return nil, nil
	}

	insertBeforeStmt := analysisinternal.StmtToInsertVarBefore(path)
	if insertBeforeStmt == nil {
		return nil, nil
	}

	tok := fset.File(expr.Pos())
	if tok == nil {
		return nil, nil
	}
	indent := calculateIndentation(src, tok, insertBeforeStmt)
	return &analysis.SuggestedFix{
		TextEdits: []analysis.TextEdit{
			{
				Pos:     insertBeforeStmt.Pos(),
				End:     insertBeforeStmt.End(),
				NewText: []byte(assignment + "\n" + indent),
			},
			{
				Pos:     rng.Start,
				End:     rng.Start,
				NewText: []byte(name),
			},
		},
	}, nil
}

// canExtractVariable reports whether the code in the given range can be
// extracted to a variable.
func canExtractVariable(fset *token.FileSet, rng span.Range, src []byte, file *ast.File, pkg *types.Package, info *types.Info) (ast.Expr, []ast.Node, bool, error) {
	if rng.Start == rng.End {
		return nil, nil, false, fmt.Errorf("start and end are equal")
	}
	path, _ := astutil.PathEnclosingInterval(file, rng.Start, rng.End)
	if len(path) == 0 {
		return nil, nil, false, fmt.Errorf("no path enclosing interval")
	}
	node := path[0]
	if rng.Start != node.Pos() || rng.End != node.End() {
		return nil, nil, false, fmt.Errorf("range does not map to an AST node")
	}
	expr, ok := node.(ast.Expr)
	if !ok {
		return nil, nil, false, fmt.Errorf("node is not an expression")
	}
	return expr, path, true, nil
}

// Calculate indentation for insertion.
// When inserting lines of code, we must ensure that the lines have consistent
// formatting (i.e. the proper indentation). To do so, we observe the indentation on the
// line of code on which the insertion occurs.
func calculateIndentation(content []byte, tok *token.File, insertBeforeStmt ast.Node) string {
	line := tok.Line(insertBeforeStmt.Pos())
	lineOffset := tok.Offset(tok.LineStart(line))
	stmtOffset := tok.Offset(insertBeforeStmt.Pos())
	return string(content[lineOffset:stmtOffset])
}

// generateAvailableIdentifier adjusts the new function name until there are no collisons in scope.
// Possible collisions include other function and variable names.
func generateAvailableIdentifier(pos token.Pos, file *ast.File, path []ast.Node, info *types.Info, prefix string, idx int) string {
	scopes := collectScopes(info, path, pos)
	name := prefix + fmt.Sprintf("%d", idx)
	for file.Scope.Lookup(name) != nil || !isValidName(name, scopes) {
		idx++
		name = fmt.Sprintf("%v%d", prefix, idx)
	}
	return name
}

// isValidName checks for variable collision in scope.
func isValidName(name string, scopes []*types.Scope) bool {
	for _, scope := range scopes {
		if scope == nil {
			continue
		}
		if scope.Lookup(name) != nil {
			return false
		}
	}
	return true
}

// returnVariable keeps track of the information we need to properly introduce a new variable
// that we will return in the extracted function.
type returnVariable struct {
	// name is the identifier that is used on the left-hand side of the call to
	// the extracted function.
	name ast.Expr
	// decl is the declaration of the variable. It is used in the type signature of the
	// extracted function and for variable declarations.
	decl *ast.Field
	// zeroVal is the "zero value" of the type of the variable. It is used in a return
	// statement in the extracted function.
	zeroVal ast.Expr
}

// extractFunction refactors the selected block of code into a new function.
// It also replaces the selected block of code with a call to the extracted
// function. First, we manually adjust the selection range. We remove trailing
// and leading whitespace characters to ensure the range is precisely bounded
// by AST nodes. Next, we determine the variables that will be the paramters
// and return values of the extracted function. Lastly, we construct the call
// of the function and insert this call as well as the extracted function into
// their proper locations.
func extractFunction(fset *token.FileSet, rng span.Range, src []byte, file *ast.File, pkg *types.Package, info *types.Info) (*analysis.SuggestedFix, error) {
	tok := fset.File(file.Pos())
	if tok == nil {
		return nil, fmt.Errorf("extractFunction: no token.File")
	}
	rng = adjustRangeForWhitespace(rng, tok, src)
	path, _ := astutil.PathEnclosingInterval(file, rng.Start, rng.End)
	if len(path) == 0 {
		return nil, fmt.Errorf("extractFunction: no path enclosing interval")
	}
	// Node that encloses the selection must be a statement.
	// TODO: Support function extraction for an expression.
	if _, ok := path[0].(ast.Stmt); !ok {
		return nil, fmt.Errorf("extractFunction: ast.Node is not a statement")
	}
	fileScope := info.Scopes[file]
	if fileScope == nil {
		return nil, fmt.Errorf("extractFunction: file scope is empty")
	}
	pkgScope := fileScope.Parent()
	if pkgScope == nil {
		return nil, fmt.Errorf("extractFunction: package scope is empty")
	}
	// Find the function declaration that encloses the selection.
	var outer *ast.FuncDecl
	for _, p := range path {
		if p, ok := p.(*ast.FuncDecl); ok {
			outer = p
			break
		}
	}
	if outer == nil {
		return nil, fmt.Errorf("extractFunction: no enclosing function")
	}

	// Find the nodes at the start and end of the selection.
	var start, end ast.Node
	ast.Inspect(outer, func(n ast.Node) bool {
		if n == nil {
			return true
		}
		// Do not override 'start' with a node that begins at the same location but is
		// nested further from 'outer'.
		if start == nil && n.Pos() == rng.Start && n.End() <= rng.End {
			start = n
		}
		if end == nil && n.End() == rng.End && n.Pos() >= rng.Start {
			end = n
		}
		return n.Pos() <= rng.End
	})
	if start == nil || end == nil {
		return nil, nil
	}

	// TODO: Support non-nested return statements.
	// A return statement is non-nested if its parent node is equal to the parent node
	// of the first node in the selection. These cases must be handled seperately because
	// non-nested return statements are guaranteed to execute. Our control flow does not
	// properly consider these situations yet.
	var retStmts []*ast.ReturnStmt
	var hasNonNestedReturn bool
	startParent := findParent(outer, start)
	ast.Inspect(outer, func(n ast.Node) bool {
		if n == nil {
			return true
		}
		if n.Pos() < rng.Start || n.End() > rng.End {
			return n.Pos() <= rng.End
		}
		ret, ok := n.(*ast.ReturnStmt)
		if !ok {
			return true
		}
		if findParent(outer, n) == startParent {
			hasNonNestedReturn = true
			return false
		}
		retStmts = append(retStmts, ret)
		return true
	})
	if hasNonNestedReturn {
		return nil, fmt.Errorf("extractFunction: selected bloc kcontains non-nested return")
	}
	containsReturnStatement := len(retStmts) > 0

	// Now that we have determined the correct range for the selection block,
	// we must determine the signature of the extracted function. We will then replace
	// the block with an assignment statement that calls the extracted function with
	// the appropriate parameters and return values.
	free, vars, assigned := collectFreeVars(info, file, fileScope, pkgScope, rng, path[0])

	var (
		params, returns         []ast.Expr     // used when calling the extracted function
		paramTypes, returnTypes []*ast.Field   // used in the signature of the extracted function
		uninitialized           []types.Object // vars we will need to initialize before the call
	)

	// Avoid duplicates while traversing vars and uninitialzed.
	seenVars := make(map[types.Object]ast.Expr)
	seenUninitialized := make(map[types.Object]struct{})

	// Each identifier in the selected block must become (1) a parameter to the
	// extracted function, (2) a return value of the extracted function, or (3) a local
	// variable in the extracted function. Determine the outcome(s) for each variable
	// based on whether it is free, altered within the selected block, and used outside
	// of the selected block.
	for _, obj := range vars {
		if _, ok := seenVars[obj]; ok {
			continue
		}
		typ := analysisinternal.TypeExpr(fset, file, pkg, obj.Type())
		if typ == nil {
			return nil, fmt.Errorf("nil AST expression for type: %v", obj.Name())
		}
		seenVars[obj] = typ
		identifier := ast.NewIdent(obj.Name())
		// An identifier must meet two conditions to become a return value of the
		// extracted function. (1) it must be used at least once after the
		// selection (isUsed), and (2) its value must be initialized or reassigned
		// within the selection (isAssigned).
		isUsed := objUsed(obj, info, rng.End, obj.Parent().End())
		_, isAssigned := assigned[obj]
		_, isFree := free[obj]
		if isUsed && isAssigned {
			returnTypes = append(returnTypes, &ast.Field{Type: typ})
			returns = append(returns, identifier)
			if !isFree {
				uninitialized = append(uninitialized, obj)
			}
		}
		// All free variables are parameters of and passed as arguments to the
		// extracted function.
		if isFree {
			params = append(params, identifier)
			paramTypes = append(paramTypes, &ast.Field{
				Names: []*ast.Ident{identifier},
				Type:  typ,
			})
		}
	}

	// Find the function literal that encloses the selection. The enclosing function literal
	// may not be the enclosing function declaration (i.e. 'outer'). For example, in the
	// following block:
	//
	// func main() {
	//     ast.Inspect(node, func(n ast.Node) bool {
	//         v := 1 // this line extracted
	//         return true
	//     })
	// }
	//
	// 'outer' is main(). However, the extracted selection most directly belongs to
	// the anonymous function literal, the second argument of ast.Inspect(). We use the
	// enclosing function literal to determine the proper return types for return statements
	// within the selection. We still need the enclosing function declaration because this is
	// the top-level declaration. We inspect the top-level declaration to look for variables
	// as well as for code replacement.
	enclosing := outer.Type
	for _, p := range path {
		if p == enclosing {
			break
		}
		if fl, ok := p.(*ast.FuncLit); ok {
			enclosing = fl.Type
			break
		}
	}

	// We put the selection in a constructed file. We can then traverse and edit
	// the extracted selection without modifying the original AST.
	startOffset := tok.Offset(rng.Start)
	endOffset := tok.Offset(rng.End)
	selection := src[startOffset:endOffset]
	extractedBlock, err := parseBlockStmt(fset, selection)
	if err != nil {
		return nil, err
	}

	// We need to account for return statements in the selected block, as they will complicate
	// the logical flow of the extracted function. See the following example, where ** denotes
	// the range to be extracted.
	//
	// Before:
	//
	// func _() int {
	//     a := 1
	//     b := 2
	//     **if a == b {
	//         return a
	//     }**
	//     ...
	// }
	//
	// After:
	//
	// func _() int {
	//     a := 1
	//     b := 2
	//     cond0, ret0 := x0(a, b)
	//     if cond0 {
	//         return ret0
	//     }
	//     ...
	// }
	//
	// func x0(a int, b int) (bool, int) {
	//     if a == b {
	//         return true, a
	//     }
	//     return false, 0
	// }
	//
	// We handle returns by adding an additional boolean return value to the extracted function.
	// This bool reports whether the original function would have returned. Because the
	// extracted selection contains a return statement, we must also add the types in the
	// return signature of the enclosing function to the return signature of the
	// extracted function. We then add an extra if statement checking this boolean value
	// in the original function. If the condition is met, the original function should
	// return a value, mimicking the functionality of the original return statement(s)
	// in the selection.

	var retVars []*returnVariable
	var ifReturn *ast.IfStmt
	if containsReturnStatement {
		// The selected block contained return statements, so we have to modify the
		// signature of the extracted function as described above. Adjust all of
		// the return statements in the extracted function to reflect this change in
		// signature.
		if err := adjustReturnStatements(returnTypes, seenVars, fset, file,
			pkg, extractedBlock); err != nil {
			return nil, err
		}
		// Collect the additional return values and types needed to accomodate return
		// statements in the selection. Update the type signature of the extracted
		// function and construct the if statement that will be inserted in the enclosing
		// function.
		retVars, ifReturn, err = generateReturnInfo(
			enclosing, pkg, path, file, info, fset, rng.Start)
		if err != nil {
			return nil, err
		}
	}

	// Add a return statement to the end of the new function. This return statement must include
	// the values for the types of the original extracted function signature and (if a return
	// statement is present in the selection) enclosing function signature.
	hasReturnValues := len(returns)+len(retVars) > 0
	if hasReturnValues {
		extractedBlock.List = append(extractedBlock.List, &ast.ReturnStmt{
			Results: append(returns, getZeroVals(retVars)...),
		})
	}

	// Construct the appropriate call to the extracted function.
	funName := generateAvailableIdentifier(rng.Start, file, path, info, "fn", 0)
	// If none of the variables on the left-hand side of the function call have
	// been initialized before the selection, we can use ':=' instead of '='.
	sym := token.ASSIGN
	if len(uninitialized) == len(returns) {
		sym = token.DEFINE
	}
	extractedFunCall := generateFuncCall(hasReturnValues, params,
		append(returns, getNames(retVars)...), funName, sym)

	// Build the extracted function.
	newFunc := &ast.FuncDecl{
		Name: ast.NewIdent(funName),
		Type: &ast.FuncType{
			Params:  &ast.FieldList{List: paramTypes},
			Results: &ast.FieldList{List: append(returnTypes, getDecls(retVars)...)},
		},
		Body: extractedBlock,
	}

	// Create variable declarations for any identifiers that need to be initialized prior to
	// calling the extracted function.
	declarations, err := initializeVars(
		uninitialized, returns, retVars, seenUninitialized, seenVars)
	if err != nil {
		return nil, err
	}

	var declBuf, replaceBuf, newFuncBuf, ifBuf bytes.Buffer
	if err := format.Node(&declBuf, fset, declarations); err != nil {
		return nil, err
	}
	if err := format.Node(&replaceBuf, fset, extractedFunCall); err != nil {
		return nil, err
	}
	if ifReturn != nil {
		if err := format.Node(&ifBuf, fset, ifReturn); err != nil {
			return nil, err
		}
	}
	if err := format.Node(&newFuncBuf, fset, newFunc); err != nil {
		return nil, err
	}

	// We're going to replace the whole enclosing function,
	// so preserve the text before and after the selected block.
	outerStart := tok.Offset(outer.Pos())
	outerEnd := tok.Offset(outer.End())
	before := src[outerStart:startOffset]
	after := src[endOffset:outerEnd]
	newLineIndent := "\n" + calculateIndentation(src, tok, start)

	var fullReplacement strings.Builder
	fullReplacement.Write(before)
	if declBuf.Len() > 0 { // add any initializations, if needed
		initializations := strings.ReplaceAll(declBuf.String(), "\n", newLineIndent) +
			newLineIndent
		fullReplacement.WriteString(initializations)
	}
	fullReplacement.Write(replaceBuf.Bytes()) // call the extracted function
	if ifBuf.Len() > 0 {                      // add the if statement below the function call, if needed
		ifstatement := newLineIndent +
			strings.ReplaceAll(ifBuf.String(), "\n", newLineIndent)
		fullReplacement.WriteString(ifstatement)
	}
	fullReplacement.Write(after)
	fullReplacement.WriteString("\n\n")       // add newlines after the enclosing function
	fullReplacement.Write(newFuncBuf.Bytes()) // insert the extracted function

	return &analysis.SuggestedFix{
		TextEdits: []analysis.TextEdit{
			{
				Pos:     outer.Pos(),
				End:     outer.End(),
				NewText: []byte(fullReplacement.String()),
			},
		},
	}, nil
}

// adjustRangeForWhitespace adjusts the given range to exclude unnecessary leading or
// trailing whitespace characters from selection. In the following example, each line
// of the if statement is indented once. There are also two extra spaces after the
// closing bracket before the line break.
//
// \tif (true) {
// \t    _ = 1
// \t}  \n
//
// By default, a valid range begins at 'if' and ends at the first whitespace character
// after the '}'. But, users are likely to highlight full lines rather than adjusting
// their cursors for whitespace. To support this use case, we must manually adjust the
// ranges to match the correct AST node. In this particular example, we would adjust
// rng.Start forward by one byte, and rng.End backwards by two bytes.
func adjustRangeForWhitespace(rng span.Range, tok *token.File, content []byte) span.Range {
	offset := tok.Offset(rng.Start)
	for offset < len(content) {
		if !unicode.IsSpace(rune(content[offset])) {
			break
		}
		// Move forwards one byte to find a non-whitespace character.
		offset += 1
	}
	rng.Start = tok.Pos(offset)

	offset = tok.Offset(rng.End)
	for offset-1 >= 0 {
		if !unicode.IsSpace(rune(content[offset-1])) {
			break
		}
		// Move backwards one byte to find a non-whitespace character.
		offset -= 1
	}
	rng.End = tok.Pos(offset)
	return rng
}

// findParent finds the parent AST node of the given target node, if the target is a
// descendant of the starting node.
func findParent(start ast.Node, target ast.Node) ast.Node {
	var parent ast.Node
	analysisinternal.WalkASTWithParent(start, func(n, p ast.Node) bool {
		if n == target {
			parent = p
			return false
		}
		return true
	})
	return parent
}

// collectFreeVars maps each identifier in the given range to whether it is "free."
// Given a range, a variable in that range is defined as "free" if it is declared
// outside of the range and neither at the file scope nor package scope. These free
// variables will be used as arguments in the extracted function. It also returns a
// list of identifiers that may need to be returned by the extracted function.
// Some of the code in this function has been adapted from tools/cmd/guru/freevars.go.
func collectFreeVars(info *types.Info, file *ast.File, fileScope *types.Scope,
	pkgScope *types.Scope, rng span.Range, node ast.Node) (map[types.Object]struct{}, []types.Object, map[types.Object]struct{}) {
	// id returns non-nil if n denotes an object that is referenced by the span
	// and defined either within the span or in the lexical environment. The bool
	// return value acts as an indicator for where it was defined.
	id := func(n *ast.Ident) (types.Object, bool) {
		obj := info.Uses[n]
		if obj == nil {
			return info.Defs[n], false
		}
		if _, ok := obj.(*types.PkgName); ok {
			return nil, false // imported package
		}
		if !(file.Pos() <= obj.Pos() && obj.Pos() <= file.End()) {
			return nil, false // not defined in this file
		}
		scope := obj.Parent()
		if scope == nil {
			return nil, false // e.g. interface method, struct field
		}
		if scope == fileScope || scope == pkgScope {
			return nil, false // defined at file or package scope
		}
		if rng.Start <= obj.Pos() && obj.Pos() <= rng.End {
			return obj, false // defined within selection => not free
		}
		return obj, true
	}
	// sel returns non-nil if n denotes a selection o.x.y that is referenced by the
	// span and defined either within the span or in the lexical environment. The bool
	// return value acts as an indicator for where it was defined.
	var sel func(n *ast.SelectorExpr) (types.Object, bool)
	sel = func(n *ast.SelectorExpr) (types.Object, bool) {
		switch x := astutil.Unparen(n.X).(type) {
		case *ast.SelectorExpr:
			return sel(x)
		case *ast.Ident:
			return id(x)
		}
		return nil, false
	}
	free := make(map[types.Object]struct{})
	var vars []types.Object
	ast.Inspect(node, func(n ast.Node) bool {
		if n == nil {
			return true
		}
		if rng.Start <= n.Pos() && n.End() <= rng.End {
			var obj types.Object
			var isFree, prune bool
			switch n := n.(type) {
			case *ast.Ident:
				obj, isFree = id(n)
			case *ast.SelectorExpr:
				obj, isFree = sel(n)
				prune = true
			}
			if obj != nil && obj.Name() != "_" {
				if isFree {
					free[obj] = struct{}{}
				}
				vars = append(vars, obj)
				if prune {
					return false
				}
			}
		}
		return n.Pos() <= rng.End
	})

	// Find identifiers that are initialized or whose values are altered at some
	// point in the selected block. For example, in a selected block from lines 2-4,
	// variables x, y, and z are included in assigned. However, in a selected block
	// from lines 3-4, only variables y and z are included in assigned.
	//
	// 1: var a int
	// 2: var x int
	// 3: y := 3
	// 4: z := x + a
	//
	assigned := make(map[types.Object]struct{})
	ast.Inspect(node, func(n ast.Node) bool {
		if n == nil {
			return true
		}
		if n.Pos() < rng.Start || n.End() > rng.End {
			return n.Pos() <= rng.End
		}
		switch n := n.(type) {
		case *ast.AssignStmt:
			for _, assignment := range n.Lhs {
				if assignment, ok := assignment.(*ast.Ident); ok {
					obj, _ := id(assignment)
					if obj == nil {
						continue
					}
					assigned[obj] = struct{}{}
				}
			}
			return false
		case *ast.DeclStmt:
			gen, ok := n.Decl.(*ast.GenDecl)
			if !ok {
				return true
			}
			for _, spec := range gen.Specs {
				vSpecs, ok := spec.(*ast.ValueSpec)
				if !ok {
					continue
				}
				for _, vSpec := range vSpecs.Names {
					obj, _ := id(vSpec)
					if obj == nil {
						continue
					}
					assigned[obj] = struct{}{}
				}
			}
			return false
		}
		return true
	})
	return free, vars, assigned
}

// canExtractFunction reports whether the code in the given range can be
// extracted to a function.
// TODO(rstambler): De-duplicate the logic between extractFunction and
// canExtractFunction.
func canExtractFunction(fset *token.FileSet, rng span.Range, src []byte, file *ast.File, pkg *types.Package, info *types.Info) bool {
	if rng.Start == rng.End {
		return false
	}
	tok := fset.File(file.Pos())
	if tok == nil {
		return false
	}
	rng = adjustRangeForWhitespace(rng, tok, src)
	path, _ := astutil.PathEnclosingInterval(file, rng.Start, rng.End)
	if len(path) == 0 {
		return false
	}
	_, ok := path[0].(ast.Stmt)
	return ok
}

// objUsed checks if the object is used between the given positions.
func objUsed(obj types.Object, info *types.Info, endSel token.Pos, endScope token.Pos) bool {
	for id, ob := range info.Uses {
		if obj == ob && endSel < id.Pos() && id.End() <= endScope {
			return true
		}
	}
	return false
}

// parseExtraction generates an AST file from the given text. We then return the portion of the
// file that represents the text.
func parseBlockStmt(fset *token.FileSet, src []byte) (*ast.BlockStmt, error) {
	text := "package main\nfunc _() { " + string(src) + " }"
	extract, err := parser.ParseFile(fset, "", text, 0)
	if err != nil {
		return nil, err
	}
	if len(extract.Decls) == 0 {
		return nil, fmt.Errorf("parsed file does not contain any declarations")
	}
	decl, ok := extract.Decls[0].(*ast.FuncDecl)
	if !ok {
		return nil, fmt.Errorf("parsed file does not contain expected function declaration")
	}
	if decl.Body == nil {
		return nil, fmt.Errorf("extracted function has no body")
	}
	return decl.Body, nil
}

// generateReturnInfo generates the information we need to adjust the return statements and
// signature of the extracted function. We prepare names, signatures, and "zero values" that
// represent the new variables. We also use this information to construct the if statement that
// is inserted below the call to the extracted function.
func generateReturnInfo(enclosing *ast.FuncType, pkg *types.Package, path []ast.Node, file *ast.File, info *types.Info, fset *token.FileSet, pos token.Pos) ([]*returnVariable, *ast.IfStmt, error) {
	// Generate information for the added bool value.
	cond := &ast.Ident{Name: generateAvailableIdentifier(pos, file, path, info, "cond", 0)}
	retVars := []*returnVariable{
		{
			name:    cond,
			decl:    &ast.Field{Type: ast.NewIdent("bool")},
			zeroVal: ast.NewIdent("false"),
		},
	}
	// Generate information for the values in the return signature of the enclosing function.
	if enclosing.Results != nil {
		for i, field := range enclosing.Results.List {
			typ := info.TypeOf(field.Type)
			if typ == nil {
				return nil, nil, fmt.Errorf(
					"failed type conversion, AST expression: %T", field.Type)
			}
			expr := analysisinternal.TypeExpr(fset, file, pkg, typ)
			if expr == nil {
				return nil, nil, fmt.Errorf("nil AST expression")
			}
			retVars = append(retVars, &returnVariable{
				name: ast.NewIdent(generateAvailableIdentifier(pos, file,
					path, info, "ret", i)),
				decl: &ast.Field{Type: expr},
				zeroVal: analysisinternal.ZeroValue(
					fset, file, pkg, typ),
			})
		}
	}
	// Create the return statement for the enclosing function. We must exclude the variable
	// for the condition of the if statement (cond) from the return statement.
	ifReturn := &ast.IfStmt{
		Cond: cond,
		Body: &ast.BlockStmt{
			List: []ast.Stmt{&ast.ReturnStmt{Results: getNames(retVars)[1:]}},
		},
	}
	return retVars, ifReturn, nil
}

// adjustReturnStatements adds "zero values" of the given types to each return statement
// in the given AST node.
func adjustReturnStatements(returnTypes []*ast.Field, seenVars map[types.Object]ast.Expr, fset *token.FileSet, file *ast.File, pkg *types.Package, extractedBlock *ast.BlockStmt) error {
	var zeroVals []ast.Expr
	// Create "zero values" for each type.
	for _, returnType := range returnTypes {
		var val ast.Expr
		for obj, typ := range seenVars {
			if typ != returnType.Type {
				continue
			}
			val = analysisinternal.ZeroValue(fset, file, pkg, obj.Type())
			break
		}
		if val == nil {
			return fmt.Errorf(
				"could not find matching AST expression for %T", returnType.Type)
		}
		zeroVals = append(zeroVals, val)
	}
	// Add "zero values" to each return statement.
	// The bool reports whether the enclosing function should return after calling the
	// extracted function. We set the bool to 'true' because, if these return statements
	// execute, the extracted function terminates early, and the enclosing function must
	// return as well.
	zeroVals = append(zeroVals, ast.NewIdent("true"))
	ast.Inspect(extractedBlock, func(n ast.Node) bool {
		if n == nil {
			return true
		}
		if n, ok := n.(*ast.ReturnStmt); ok {
			n.Results = append(zeroVals, n.Results...)
			return true
		}
		return true
	})
	return nil
}

// generateFuncCall constructs a call expression for the extracted function, described by the
// given parameters and return variables.
func generateFuncCall(hasReturnVals bool, params, returns []ast.Expr, name string, token token.Token) ast.Node {
	var replace ast.Node
	if hasReturnVals {
		callExpr := &ast.CallExpr{
			Fun:  ast.NewIdent(name),
			Args: params,
		}
		replace = &ast.AssignStmt{
			Lhs: returns,
			Tok: token,
			Rhs: []ast.Expr{callExpr},
		}
	} else {
		replace = &ast.CallExpr{
			Fun:  ast.NewIdent(name),
			Args: params,
		}
	}
	return replace
}

// initializeVars creates variable declarations, if needed.
// Our preference is to replace the selected block with an "x, y, z := fn()" style
// assignment statement. We can use this style when none of the variables in the
// extracted function's return statement have already be initialized outside of the
// selected block. However, for example, if z is already defined elsewhere, we
// replace the selected block with:
//
// var x int
// var y string
// x, y, z = fn()
func initializeVars(uninitialized []types.Object, returns []ast.Expr, retVars []*returnVariable, seenUninitialized map[types.Object]struct{}, seenVars map[types.Object]ast.Expr) ([]ast.Stmt, error) {
	var declarations []ast.Stmt
	// We do not manually initialize variables if every return value is unitialized.
	// We can use := to initialize the variables in this situation.
	if len(uninitialized) == len(returns) {
		return declarations, nil
	}
	for _, obj := range uninitialized {
		if _, ok := seenUninitialized[obj]; ok {
			continue
		}
		seenUninitialized[obj] = struct{}{}
		valSpec := &ast.ValueSpec{
			Names: []*ast.Ident{ast.NewIdent(obj.Name())},
			Type:  seenVars[obj],
		}
		genDecl := &ast.GenDecl{
			Tok:   token.VAR,
			Specs: []ast.Spec{valSpec},
		}
		declarations = append(declarations, &ast.DeclStmt{Decl: genDecl})
	}
	// Each variable added from a return statement in the selection
	// must be initialized.
	for i, retVar := range retVars {
		n := retVar.name.(*ast.Ident)
		valSpec := &ast.ValueSpec{
			Names: []*ast.Ident{n},
			Type:  retVars[i].decl.Type,
		}
		genDecl := &ast.GenDecl{
			Tok:   token.VAR,
			Specs: []ast.Spec{valSpec},
		}
		declarations = append(declarations, &ast.DeclStmt{Decl: genDecl})
	}
	return declarations, nil
}

// getNames returns the names from the given list of returnVariable.
func getNames(retVars []*returnVariable) []ast.Expr {
	var names []ast.Expr
	for _, retVar := range retVars {
		names = append(names, retVar.name)
	}
	return names
}

// getZeroVals returns the "zero values" from the given list of returnVariable.
func getZeroVals(retVars []*returnVariable) []ast.Expr {
	var zvs []ast.Expr
	for _, retVar := range retVars {
		zvs = append(zvs, retVar.zeroVal)
	}
	return zvs
}

// getDecls returns the declarations from the given list of returnVariable.
func getDecls(retVars []*returnVariable) []*ast.Field {
	var decls []*ast.Field
	for _, retVar := range retVars {
		decls = append(decls, retVar.decl)
	}
	return decls
}
