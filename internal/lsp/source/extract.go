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
	if rng.Start == rng.End {
		return nil, fmt.Errorf("extractVariable: start and end are equal (%v)", fset.Position(rng.Start))
	}
	path, _ := astutil.PathEnclosingInterval(file, rng.Start, rng.End)
	if len(path) == 0 {
		return nil, fmt.Errorf("extractVariable: no path enclosing interval")
	}
	node := path[0]
	if rng.Start != node.Pos() || rng.End != node.End() {
		return nil, fmt.Errorf("extractVariable: node doesn't perfectly enclose range")
	}
	expr, ok := node.(ast.Expr)
	if !ok {
		return nil, fmt.Errorf("extractVariable: node is not an expression")
	}
	name := generateAvailableIdentifier(expr.Pos(), file, path, info)

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

	tok := fset.File(node.Pos())
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
// TODO(rstambler): De-duplicate the logic between extractVariable and
// canExtractVariable.
func canExtractVariable(fset *token.FileSet, rng span.Range, src []byte, file *ast.File, pkg *types.Package, info *types.Info) bool {
	if rng.Start == rng.End {
		return false
	}
	path, _ := astutil.PathEnclosingInterval(file, rng.Start, rng.End)
	if len(path) == 0 {
		return false
	}
	node := path[0]
	if rng.Start != node.Pos() || rng.End != node.End() {
		return false
	}
	_, ok := node.(ast.Expr)
	return ok
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

// Check for variable collision in scope.
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
	// Node that encloses selection must be a statement.
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
	// Find function enclosing the selection.
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
	// At the moment, we don't extract selections containing return statements,
	// as they are more complex and need to be adjusted to maintain correctness.
	// TODO: Support extracting and rewriting code with return statements.
	var containsReturn bool
	ast.Inspect(outer, func(n ast.Node) bool {
		if n == nil {
			return true
		}
		if rng.Start <= n.Pos() && n.End() <= rng.End {
			if _, ok := n.(*ast.ReturnStmt); ok {
				containsReturn = true
				return false
			}
		}
		return n.Pos() <= rng.End
	})
	if containsReturn {
		return nil, fmt.Errorf("extractFunction: selected block contains return")
	}
	// Find the nodes at the start and end of the selection.
	var start, end ast.Node
	ast.Inspect(outer, func(n ast.Node) bool {
		if n == nil {
			return true
		}
		if n.Pos() == rng.Start && n.End() <= rng.End {
			start = n
		}
		if n.End() == rng.End && n.Pos() >= rng.Start {
			end = n
		}
		return n.Pos() <= rng.End
	})
	if start == nil || end == nil {
		return nil, fmt.Errorf("extractFunction: start or end node is empty")
	}

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

	// Our preference is to replace the selected block with an "x, y, z := fn()" style
	// assignment statement. We can use this style when none of the variables in the
	// extracted function's return statement have already be initialized outside of the
	// selected block. However, for example, if z is already defined elsewhere, we
	// replace the selected block with:
	//
	// var x int
	// var y string
	// x, y, z = fn()
	//
	var initializations string
	if len(uninitialized) > 0 && len(uninitialized) != len(returns) {
		var declarations []ast.Stmt
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
		var declBuf bytes.Buffer
		if err := format.Node(&declBuf, fset, declarations); err != nil {
			return nil, err
		}
		indent := calculateIndentation(src, tok, start)
		// Add proper indentation to each declaration. Also add formatting to
		// the line following the last initialization to ensure that subsequent
		// edits begin at the proper location.
		initializations = strings.ReplaceAll(declBuf.String(), "\n", "\n"+indent) +
			"\n" + indent
	}

	name := generateAvailableIdentifier(start.Pos(), file, path, info)
	var replace ast.Node
	if len(returns) > 0 {
		// If none of the variables on the left-hand side of the function call have
		// been initialized before the selection, we can use := instead of =.
		assignTok := token.ASSIGN
		if len(uninitialized) == len(returns) {
			assignTok = token.DEFINE
		}
		callExpr := &ast.CallExpr{
			Fun:  ast.NewIdent(name),
			Args: params,
		}
		replace = &ast.AssignStmt{
			Lhs: returns,
			Tok: assignTok,
			Rhs: []ast.Expr{callExpr},
		}
	} else {
		replace = &ast.CallExpr{
			Fun:  ast.NewIdent(name),
			Args: params,
		}
	}

	startOffset := tok.Offset(rng.Start)
	endOffset := tok.Offset(rng.End)
	selection := src[startOffset:endOffset]
	// Put selection in constructed file to parse and produce block statement. We can
	// then use the block statement to traverse and edit extracted function without
	// altering the original file.
	text := "package main\nfunc _() { " + string(selection) + " }"
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
	// Add return statement to the end of the new function.
	if len(returns) > 0 {
		decl.Body.List = append(decl.Body.List,
			&ast.ReturnStmt{Results: returns},
		)
	}
	funcDecl := &ast.FuncDecl{
		Name: ast.NewIdent(name),
		Type: &ast.FuncType{
			Params:  &ast.FieldList{List: paramTypes},
			Results: &ast.FieldList{List: returnTypes},
		},
		Body: decl.Body,
	}

	var replaceBuf, newFuncBuf bytes.Buffer
	if err := format.Node(&replaceBuf, fset, replace); err != nil {
		return nil, err
	}
	if err := format.Node(&newFuncBuf, fset, funcDecl); err != nil {
		return nil, err
	}

	outerStart := tok.Offset(outer.Pos())
	outerEnd := tok.Offset(outer.End())
	// We're going to replace the whole enclosing function,
	// so preserve the text before and after the selected block.
	before := src[outerStart:startOffset]
	after := src[endOffset:outerEnd]
	var fullReplacement strings.Builder
	fullReplacement.Write(before)
	fullReplacement.WriteString(initializations) // add any initializations, if needed
	fullReplacement.Write(replaceBuf.Bytes())    // call the extracted function
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

// Adjust new function name until no collisons in scope. Possible collisions include
// other function and variable names.
func generateAvailableIdentifier(pos token.Pos, file *ast.File, path []ast.Node, info *types.Info) string {
	scopes := collectScopes(info, path, pos)
	var idx int
	name := "x0"
	for file.Scope.Lookup(name) != nil || !isValidName(name, scopes) {
		idx++
		name = fmt.Sprintf("x%d", idx)
	}
	return name
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

// objUsed checks if the object is used after the selection but within
// the scope of the enclosing function.
func objUsed(obj types.Object, info *types.Info, endSel token.Pos, endScope token.Pos) bool {
	for id, ob := range info.Uses {
		if obj == ob && endSel < id.Pos() && id.End() <= endScope {
			return true
		}
	}
	return false
}
