// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"log"
	"os"
	"strings"

	"golang.org/x/tools/go/ast/astutil"

	internalastutil "runtime/_mkmalloc/astutil"
)

var stdout = flag.Bool("stdout", false, "write sizeclasses source to stdout instead of sizeclasses.go")

func makeSizeToSizeClass(classes []class) []uint8 {
	sc := uint8(0)
	ret := make([]uint8, benchmarkMax+1)
	for i := range ret {
		if i > classes[sc].size {
			sc++
		}
		ret[i] = sc
	}
	return ret
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("mkmalloc: ")

	classes := makeClasses()
	sizeToSizeClass := makeSizeToSizeClass(classes)

	if *stdout {
		if _, err := os.Stdout.Write(mustFormat(generateSizeClasses(classes))); err != nil {
			log.Fatal(err)
		}
		return
	}

	sizeclasesesfile := "../../internal/runtime/gc/sizeclasses.go"
	if err := os.WriteFile(sizeclasesesfile, mustFormat(generateSizeClasses(classes)), 0666); err != nil {
		log.Fatal(err)
	}

	outfile := "../malloc_generated.go"
	if err := os.WriteFile(outfile, mustFormat(inline(specializedMallocConfig(classes, sizeToSizeClass))), 0666); err != nil {
		log.Fatal(err)
	}

	tablefile := "../malloc_tables_generated.go"
	if err := os.WriteFile(tablefile, mustFormat(generateTable(sizeToSizeClass)), 0666); err != nil {
		log.Fatal(err)
	}

	benchmarkFile := "../malloc_bench_generated_test.go"
	if err := os.WriteFile(benchmarkFile, mustFormat(append(inline(benchmarkConfig(classes, sizeToSizeClass)), []byte(generateTopBenchmark(classes, sizeToSizeClass))...)), 0666); err != nil {
		log.Fatal(err)
	}

}

// withLineNumbers returns b with line numbers added to help debugging.
func withLineNumbers(b []byte) []byte {
	var buf bytes.Buffer
	i := 1
	for line := range bytes.Lines(b) {
		fmt.Fprintf(&buf, "%d: %s", i, line)
		i++
	}
	return buf.Bytes()
}

// mustFormat formats the input source, or exits if there's an error.
func mustFormat(b []byte) []byte {
	formatted, err := format.Source(b)
	if err != nil {
		log.Fatalf("error formatting source: %v\nsource:\n%s\n", err, withLineNumbers(b))
	}
	return formatted
}

// generatorConfig is the configuration for the generator. It uses the given file to find
// its templates, and generates each of the functions specified by specs.
type generatorConfig struct {
	file  string
	specs []spec
}

// spec is the specification for a function for the inliner to produce. The function gets
// the given name, and is produced by starting with the function with the name given by
// templateFunc and applying each of the ops.
type spec struct {
	name         string
	templateFunc string
	ops          []op
}

// replacementKind specifies the operation to ben done by a op.
type replacementKind int

const (
	inlineFunc = replacementKind(iota)
	subBasicLit
	foldCondition
	subIdent
	deleteConst
)

// op is a single inlining operation for the inliner. Any calls to the function
// from are replaced with the inlined body of to. For non-functions, uses of from are
// replaced with the basic literal expression given by to.
type op struct {
	kind replacementKind
	from string
	to   string
}

func smallScanNoHeaderSCFuncName(sc, scMax uint8) string {
	if sc == 0 || sc > scMax {
		return "mallocPanic"
	}
	return fmt.Sprintf("mallocgcSmallScanNoHeaderSC%d", sc)
}

const tinyFuncName = "mallocgcTinySC2"

func smallNoScanSCFuncName(sc, scMax uint8) string {
	if sc < 2 || sc > scMax {
		return "mallocPanic"
	}
	return fmt.Sprintf("mallocgcSmallNoScanSC%d", sc)
}

// specializedMallocConfig produces an inlining config to stamp out the definitions of the size-specialized
// malloc functions to be written by mkmalloc.
func specializedMallocConfig(classes []class, sizeToSizeClass []uint8) generatorConfig {
	config := generatorConfig{file: "../malloc_stubs.go"}

	// Only generate specialized functions for sizes up to specializedMallocMax.
	// We've noticed limited benefit (or sometimes worse performance) for specialized
	// functions for larger sizes, and having too many functions causes icache issues.
	scMax := sizeToSizeClass[specializedMallocMax]

	str := fmt.Sprint

	// allocations with pointer bits
	{
		const noscan = 0
		for sc := uint8(0); sc <= scMax; sc++ {
			if sc == 0 {
				continue
			}
			name := smallScanNoHeaderSCFuncName(sc, scMax)
			elemsize := classes[sc].size
			config.specs = append(config.specs, spec{
				templateFunc: "mallocStub",
				name:         name,
				ops: []op{
					{inlineFunc, "inlinedMalloc", "smallStub"},
					{inlineFunc, "postMallocgc", "postMallocgc"},
					{foldCondition, "isNoScan_", str(false)},
					{inlineFunc, "heapSetTypeNoHeaderStub", "heapSetTypeNoHeaderStub"},
					{inlineFunc, "nextFreeFastStub", "nextFreeFastStub"},
					{inlineFunc, "writeHeapBitsSmallStub", "writeHeapBitsSmallStub"},
					{foldCondition, "isSlowPath_", str(false)},
					{subBasicLit, "elemsize_", str(elemsize)},
					{subBasicLit, "sizeclass_", str(sc)},
					{subBasicLit, "noscanint_", str(noscan)},
					{foldCondition, "isTiny_", str(false)},
					{subIdent, "mallocgcSlowPathStub", "mallocgcSmallScanSlowPath"},
				},
			})
		}
	}

	// allocations without pointer bits
	{
		const noscan = 1

		// tiny
		tinySizeClass := sizeToSizeClass[tinySize]
		{
			name := tinyFuncName
			elemsize := classes[tinySizeClass].size
			config.specs = append(config.specs, spec{
				templateFunc: "mallocStub",
				name:         name,
				ops: []op{
					{inlineFunc, "inlinedMalloc", "tinyStub"},
					{inlineFunc, "nextFreeFastTiny", "nextFreeFastTiny"},
					{inlineFunc, "postMallocgc", "postMallocgc"},
					{inlineFunc, "nextFreeFastStub", "nextFreeFastStub"},
					{foldCondition, "isSlowPath_", str(false)},
					{subBasicLit, "elemsize_", str(elemsize)},
					{subBasicLit, "sizeclass_", str(tinySizeClass)},
					{subBasicLit, "noscanint_", str(noscan)},
					{foldCondition, "isTiny_", str(true)},
				},
			})
		}

		// non-tiny
		for sc := uint8(tinySizeClass); sc <= scMax; sc++ {
			name := smallNoScanSCFuncName(sc, scMax)
			elemsize := classes[sc].size
			config.specs = append(config.specs, spec{
				templateFunc: "mallocStub",
				name:         name,
				ops: []op{
					{inlineFunc, "inlinedMalloc", "smallStub"},
					{inlineFunc, "postMallocgc", "postMallocgc"},
					{foldCondition, "isNoScan_", str(true)},
					{inlineFunc, "nextFreeFastStub", "nextFreeFastStub"},
					{foldCondition, "isSlowPath_", str(false)},
					{subBasicLit, "elemsize_", str(elemsize)},
					{subBasicLit, "sizeclass_", str(sc)},
					{subBasicLit, "noscanint_", str(noscan)},
					{foldCondition, "isTiny_", str(false)},
					{subIdent, "mallocgcSlowPathStub", "mallocgcSmallNoScanSlowPath"},
				},
			})
		}
	}

	// Non-size-specialized fallbacks in case we can't do the fast path.
	config.specs = append(config.specs, spec{
		templateFunc: "mallocStub",
		name:         "mallocgcTinySlowPath",
		ops: []op{
			{inlineFunc, "inlinedMalloc", "tinyStub"},
			{inlineFunc, "postMallocgc", "postMallocgc"},
			{inlineFunc, "nextFreeFastTiny", "nextFreeFastTiny"},
			{inlineFunc, "deductAssistCredit", "deductAssistCredit"},
			{foldCondition, "isSlowPath_", str(true)},
			{foldCondition, "isTiny_", str(true)},
			{subBasicLit, "elemsize_", str(classes[sizeToSizeClass[tinySize]].size)},
		},
	})
	config.specs = append(config.specs, spec{
		templateFunc: "mallocgcSlowPathStub",
		name:         "mallocgcSmallScanSlowPath",
		ops: []op{
			{inlineFunc, "mallocStub", "mallocStub"},
			{inlineFunc, "inlinedMalloc", "smallStub"},
			{inlineFunc, "heapSetTypeNoHeaderStub", "heapSetTypeNoHeaderStub"},
			{inlineFunc, "writeHeapBitsSmallStub", "writeHeapBitsSmallStub"},
			{inlineFunc, "postMallocgc", "postMallocgc"},
			{inlineFunc, "nextFreeFastStub", "nextFreeFastStub"},
			{inlineFunc, "deductAssistCredit", "deductAssistCredit"},
			{foldCondition, "isSlowPath_", str(true)},
			{foldCondition, "isTiny_", str(false)},
			{foldCondition, "isNoScan_", str(false)},

			// Remove constants used by size-specialized variants.
			{deleteConst, "elemsize", ""},
			{deleteConst, "sizeclass", ""},
			{deleteConst, "spc", ""},
		},
	})
	config.specs = append(config.specs, spec{
		templateFunc: "mallocgcSlowPathStub",
		name:         "mallocgcSmallNoScanSlowPath",
		ops: []op{
			{inlineFunc, "mallocStub", "mallocStub"},
			{inlineFunc, "inlinedMalloc", "smallStub"},
			{inlineFunc, "postMallocgc", "postMallocgc"},
			{inlineFunc, "nextFreeFastStub", "nextFreeFastStub"},
			{inlineFunc, "deductAssistCredit", "deductAssistCredit"},
			{foldCondition, "isSlowPath_", str(true)},
			{foldCondition, "isTiny_", str(false)},
			{foldCondition, "isNoScan_", str(true)},

			// Remove constants used by size-specialized variants.
			{deleteConst, "elemsize", ""},
			{deleteConst, "sizeclass", ""},
			{deleteConst, "spc", ""},
		},
	})

	return config
}

// inline applies the inlining operations given by the config.
func inline(config generatorConfig) []byte {
	var out bytes.Buffer

	// Read the template file in.
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, config.file, nil, parser.SkipObjectResolution)
	if err != nil {
		log.Fatalf("parsing %s: %v", config.file, err)
	}

	// Collect the function and import declarations. The function
	// declarations in the template file provide both the templates
	// that will be stamped out, and the functions that will be inlined
	// into them. The imports from the template file will be copied
	// straight to the output.
	funcDecls := map[string]*ast.FuncDecl{}
	importDecls := []*ast.GenDecl{}
	for _, decl := range f.Decls {
		switch decl := decl.(type) {
		case *ast.FuncDecl:
			funcDecls[decl.Name.Name] = decl
		case *ast.GenDecl:
			if decl.Tok.String() == "import" {
				importDecls = append(importDecls, decl)
				continue
			}
		}
	}

	// Write out the package and import declarations.
	out.WriteString("// Code generated by mkmalloc.go; DO NOT EDIT.\n")
	out.WriteString("// See overview in malloc_stubs.go.\n\n")
	out.WriteString("package " + f.Name.Name + "\n\n")
	for _, importDecl := range importDecls {
		out.Write(mustFormatNode(fset, importDecl))
		out.WriteString("\n\n")
	}

	// Produce each of the inlined functions specified by specs.
	for _, spec := range config.specs {
		// Start with a renamed copy of the template function.
		containingFuncCopy := internalastutil.CloneNode(funcDecls[spec.templateFunc])
		if containingFuncCopy == nil {
			log.Fatal("did not find", spec.templateFunc)
		}
		containingFuncCopy.Name.Name = spec.name

		// Apply each of the ops given by the specs
		stamped := ast.Node(containingFuncCopy)
		for _, repl := range spec.ops {
			switch repl.kind {
			case inlineFunc:
				if toDecl, ok := funcDecls[repl.to]; ok {
					stamped = inlineFunction(stamped, repl.from, toDecl)
				}
			case subBasicLit:
				stamped = substituteWithBasicLit(stamped, repl.from, repl.to)
			case foldCondition:
				stamped = foldIfCondition(stamped, repl.from, repl.to)
			case subIdent:
				stamped = substituteIdent(stamped, repl.from, repl.to)
			case deleteConst:
				stamped = deleteConstDecl(stamped, repl.from)
			default:
				log.Fatalf("unknown op kind %v", repl.kind)
			}
		}

		stamped = cleanLabels(stamped)

		out.Write(mustFormatNode(fset, stamped))
		out.WriteString("\n\n")
	}

	return out.Bytes()
}

// substituteWithBasicLit recursively renames identifiers in the provided AST
// according to 'from' and 'to'.
func substituteWithBasicLit(node ast.Node, from, to string) ast.Node {
	// The op is a substitution of an identifier with a basic literal.
	toExpr, err := parser.ParseExpr(to)
	if err != nil {
		log.Fatalf("parsing expr %q: %v", to, err)
	}
	toLit, ok := toExpr.(*ast.BasicLit)
	if !ok {
		log.Fatalf("op 'to' expr %q is not a basic literal", to)
	}
	return astutil.Apply(node, func(cursor *astutil.Cursor) bool {
		if ident, ok := cursor.Node().(*ast.Ident); ok && ident.Name == from {
			replacement := *toLit
			replacement.ValuePos = ident.NamePos
			cursor.Replace(new(replacement))
		}
		return true
	}, nil)
}

// substituteIdent replaces the ident named 'from' to 'to'.
func substituteIdent(node ast.Node, from, to string) ast.Node {
	return astutil.Apply(node, func(cursor *astutil.Cursor) bool {
		if ident, ok := cursor.Node().(*ast.Ident); ok && ident.Name == from {
			cursor.Replace(&ast.Ident{Name: to, NamePos: ident.NamePos})
		}
		return true
	}, nil)
}

// foldIfCondition replaces 'from' with 'to', which must be "true" or "false".
// It then applies simplifications to any boolean expressions that have literal
// true or false values, from the bottom up. Any if statements that have a condition
// that is a literal true or false after the simplification will be replaced with
// their bodies (in the true case) or deleted (in the false case).
func foldIfCondition(node ast.Node, from, to string) ast.Node {
	boolLit := func(n ast.Expr) (v, ok bool) {
		if ident, ok := ast.Unparen(n).(*ast.Ident); ok {
			switch ident.Name {
			case "true":
				return true, true
			case "false":
				return false, true
			}
			return false, false
		}
		return false, false
	}
	handleIfs := func(cursor *astutil.Cursor) bool {
		switch n := cursor.Node().(type) {
		case *ast.Ident:
			// First, do the replacement.
			if n.Name == from {
				cursor.Replace(&ast.Ident{Name: to, NamePos: n.NamePos})
			}
		case *ast.UnaryExpr:
			if n.Op == token.NOT {
				if b, ok := boolLit(n.X); ok {
					name := "true"
					if b {
						name = "false"
					}
					cursor.Replace(&ast.Ident{Name: name, NamePos: n.Pos()})
				}
			}
		case *ast.BinaryExpr:
			xBool, xOk := boolLit(n.X)
			yBool, yOk := boolLit(n.Y)
			if n.Op == token.LAND {
				switch {
				case xOk && !xBool || yOk && !yBool:
					cursor.Replace(&ast.Ident{Name: "false", NamePos: n.Pos()})
				case xOk && xBool:
					cursor.Replace(n.Y)
				case yOk && yBool:
					cursor.Replace(n.X)
				}
			} else if n.Op == token.LOR {
				switch {
				case xOk && xBool || yOk && yBool:
					cursor.Replace(&ast.Ident{Name: "true", NamePos: n.Pos()})
				case xOk && !xBool:
					cursor.Replace(n.Y)
				case yOk && !yBool:
					cursor.Replace(n.X)
				}
			}
		case *ast.IfStmt:
			if v, ok := boolLit(n.Cond); ok {
				if cursor.Index() < 0 {
					replacement := ast.Node(&ast.EmptyStmt{})
					if v {
						replacement = n.Body
					}
					cursor.Replace(replacement)
					break
				}
				if v {
					for _, stmt := range n.Body.List {
						cursor.InsertBefore(stmt)
					}
				} else if n.Else != nil {
					if block, ok := n.Else.(*ast.BlockStmt); ok {
						for i := len(block.List) - 1; i >= 0; i-- {
							cursor.InsertAfter(block.List[i])
						}
					}
				}
				cursor.Delete()
			}
		case *ast.LabeledStmt:
			// This case isn't necessary but it moves the code
			// out of the block so that it looks cleaner.
			if inner, ok := n.Stmt.(*ast.BlockStmt); ok {
				if len(inner.List) == 0 {
					cursor.Delete()
					break
				}
				list := inner.List
				n.Stmt = list[0]
				for i := len(list) - 1; i > 0; i-- {
					cursor.InsertAfter(list[i])
				}
			}
		}
		return true
	}
	return astutil.Apply(node, nil, handleIfs)
}

func cleanLabels(node ast.Node) ast.Node {
	found := map[string]bool{}
	ast.Inspect(node, func(node ast.Node) bool {
		if branch, ok := node.(*ast.BranchStmt); ok {
			if branch.Label != nil {
				found[branch.Label.Name] = true
			}
		}
		return true
	})
	return astutil.Apply(node, nil, func(cursor *astutil.Cursor) bool {
		if lstmt, ok := cursor.Node().(*ast.LabeledStmt); ok {
			if !found[lstmt.Label.Name] {
				if _, ok := lstmt.Stmt.(*ast.EmptyStmt); ok {
					cursor.Delete()
				} else {
					cursor.Replace(lstmt.Stmt)
				}
			}
		}
		return true
	})
}

// reports whether this is a non-grouped constant decl named 'name'.
func isNamedConstDecl(node ast.Node, name string) bool {
	declStmt, ok := node.(*ast.DeclStmt)
	if !ok {
		return false
	}

	genDecl, ok := declStmt.Decl.(*ast.GenDecl)
	if !ok || genDecl.Tok != token.CONST {
		return false
	}

	if len(genDecl.Specs) != 1 {
		return false
	}
	vs, ok := genDecl.Specs[0].(*ast.ValueSpec)
	if !ok || len(vs.Names) != 1 || len(vs.Values) != 1 {
		return false
	}

	return vs.Names[0].Name == name
}

// deleteConstDecl removes const declarations whose name matches the given name.
// It only applies to declaration statements with a single declaration.
func deleteConstDecl(node ast.Node, name string) ast.Node {
	return astutil.Apply(node, func(cursor *astutil.Cursor) bool {
		if isNamedConstDecl(cursor.Node(), name) {
			cursor.Delete()
		}
		return true
	}, nil)
}

// inlineFunction recursively replaces calls to the function 'from' with the body of the function
// 'toDecl'. All calls to 'from' must either have no return values and appear in standalone expression statements
// or otherwise must appear in assignment statements.
// The replacement is very simple: it doesn't substitute the arguments for the parameters, so the
// arguments to the function call must be the same identifier as the parameters to the function
// declared by 'toDecl'. If there are any calls to from where that's not the case there will be a fatal error.
func inlineFunction(node ast.Node, from string, toDecl *ast.FuncDecl) ast.Node {
	return astutil.Apply(node, func(cursor *astutil.Cursor) bool {
		switch node := cursor.Node().(type) {
		case *ast.AssignStmt:
			// TODO(matloob) CHECK function args have same name
			// as parameters (or parameter is "_").
			if len(node.Rhs) == 1 && isCallTo(node.Rhs[0], from) {
				args := node.Rhs[0].(*ast.CallExpr).Args
				if !argsMatchParameters(args, toDecl.Type.Params) {
					log.Fatalf("applying op: arguments to %v don't match parameter names of %v: %v", from, toDecl.Name, debugPrint(args...))
				}
				replaceAssignment(cursor, node, toDecl)
			}
			return false
		case *ast.ExprStmt:
			if callExpr, ok := node.X.(*ast.CallExpr); ok && isCallTo(callExpr, from) {
				if !argsMatchParameters(callExpr.Args, toDecl.Type.Params) {
					log.Fatalf("applying op: arguments to %v don't match parameter names of %v: %v", from, toDecl.Name, debugPrint(callExpr.Args...))
				}
				if toDecl.Type.Results != nil {
					log.Fatalf("applying op: call to %v, which does not appear in an assignment, is replaced with %v which has return values: %v", from, toDecl.Name, debugPrint(callExpr.Args...))
				}
				replaceCallExprStmt(cursor, toDecl)
			}
			return false
		case *ast.ReturnStmt:
			if len(node.Results) == 1 && isCallTo(node.Results[0], from) {
				args := node.Results[0].(*ast.CallExpr).Args
				if !argsMatchParameters(args, toDecl.Type.Params) {
					log.Fatalf("applying op: arguments to %v don't match parameter names of %v: %v", from, toDecl.Name, debugPrint(args...))
				}
				replaceTailCall(cursor, toDecl)
			}
			return false
		case *ast.CallExpr:
			if isCallTo(node, from) {
				switch cursor.Parent().(type) {
				case *ast.AssignStmt, *ast.ExprStmt:
				default:
					log.Fatalf("applying op: all calls to function %q being replaced must appear in an assignment or expression statement, appears in %T", from, cursor.Parent())
				}
			}
		}
		return true
	}, nil)
}

// argsMatchParameters reports whether the arguments given by args are all identifiers
// whose names are the same as the corresponding parameters in params.
func argsMatchParameters(args []ast.Expr, params *ast.FieldList) bool {
	var paramIdents []*ast.Ident
	for _, f := range params.List {
		paramIdents = append(paramIdents, f.Names...)
	}

	if len(args) != len(paramIdents) {
		return false
	}

	for i := range args {
		if !isIdentWithName(args[i], paramIdents[i].Name) {
			return false
		}
	}

	return true
}

// isIdentWithName reports whether the expression is an identifier with the given name.
func isIdentWithName(expr ast.Node, name string) bool {
	ident, ok := expr.(*ast.Ident)
	if !ok {
		return false
	}
	return ident.Name == name
}

// isCallTo reports whether the expression is a call expression to the function with the given name.
func isCallTo(expr ast.Expr, name string) bool {
	callexpr, ok := expr.(*ast.CallExpr)
	if !ok {
		return false
	}
	return isIdentWithName(callexpr.Fun, name)
}

// replaceCallExprStmt replaces a standalone expression statement calling a function with no
// return values with the body of the function.
func replaceCallExprStmt(cursor *astutil.Cursor, funcdecl *ast.FuncDecl) {
	body := internalastutil.CloneNode(funcdecl.Body)
	for _, stmt := range body.List {
		cursor.InsertBefore(stmt)
	}
	cursor.Delete()
}

func replaceTailCall(cursor *astutil.Cursor, funcdecl *ast.FuncDecl) {
	if !hasTerminatingReturn(funcdecl.Body) {
		log.Fatal("function being inlined must have a return at the end")
	}

	body := internalastutil.CloneNode(funcdecl.Body)
	if len(body.List) < 1 {
		log.Fatal("replacing with empty bodied function")
	}

	// The op happens in two steps: first we insert the body of the function being inlined (except for
	// the final return) before the assignment, and then we change the assignment statement to replace the function call
	// with the expressions being returned.

	// Insert the body up to the final return.
	for _, stmt := range body.List {
		cursor.InsertBefore(stmt)
	}
	cursor.Delete()
}

// replaceAssignment replaces an assignment statement where the right hand side is a function call
// whose arguments have the same names as the parameters to funcdecl with the body of funcdecl.
// It sets the left hand side of the assignment to the return values of the function.
func replaceAssignment(cursor *astutil.Cursor, assign *ast.AssignStmt, funcdecl *ast.FuncDecl) {
	if !hasTerminatingReturn(funcdecl.Body) {
		log.Fatal("function being inlined must have a return at the end")
	}

	body := internalastutil.CloneNode(funcdecl.Body)
	if hasTerminatingAndNonterminatingReturn(funcdecl.Body) {
		// The function has multiple return points. Add the code that we'd continue with in the caller
		// after each of the return points. The calling function must have a terminating return
		// so we don't continue execution in the replaced function after we finish executing the
		// continue block that we add.
		body = addContinues(cursor, assign, body, everythingFollowingInParent(cursor)).(*ast.BlockStmt)
	}

	if len(body.List) < 1 {
		log.Fatal("replacing with empty bodied function")
	}

	// The op happens in two steps: first we insert the body of the function being inlined (except for
	// the final return) before the assignment, and then we change the assignment statement to replace the function call
	// with the expressions being returned.

	// Determine the expressions being returned.
	beforeReturn, ret := body.List[:len(body.List)-1], body.List[len(body.List)-1]
	returnStmt, ok := ret.(*ast.ReturnStmt)
	if !ok {
		log.Fatal("last stmt in function we're replacing with should be a return")
	}
	results := returnStmt.Results

	// Insert the body up to the final return.
	for _, stmt := range beforeReturn {
		cursor.InsertBefore(stmt)
	}

	// Rewrite the assignment statement.
	replaceWithAssignment(cursor, assign.Lhs, results, assign.Tok)
}

// hasTerminatingReturn reparts whether the block ends in a return statement.
func hasTerminatingReturn(block *ast.BlockStmt) bool {
	_, ok := block.List[len(block.List)-1].(*ast.ReturnStmt)
	return ok
}

// hasTerminatingAndNonterminatingReturn reports whether the block ends in a return
// statement, and also has a return elsewhere in it.
func hasTerminatingAndNonterminatingReturn(block *ast.BlockStmt) bool {
	if !hasTerminatingReturn(block) {
		return false
	}
	var ret bool
	for i := range block.List[:len(block.List)-1] {
		ast.Inspect(block.List[i], func(node ast.Node) bool {
			_, ok := node.(*ast.ReturnStmt)
			if ok {
				ret = true
				return false
			}
			return true
		})
	}
	return ret
}

// everythingFollowingInParent returns a block with everything in the parent block node of the cursor after
// the cursor itself. The cursor must point to an element in a block node's list.
func everythingFollowingInParent(cursor *astutil.Cursor) *ast.BlockStmt {
	parent := cursor.Parent()
	block, ok := parent.(*ast.BlockStmt)
	if !ok {
		log.Fatal("internal error: in everythingFollowingInParent, cursor doesn't point to element in block list")
	}

	blockcopy := internalastutil.CloneNode(block)      // get a clean copy
	blockcopy.List = blockcopy.List[cursor.Index()+1:] // and remove everything before and including stmt

	if _, ok := blockcopy.List[len(blockcopy.List)-1].(*ast.ReturnStmt); !ok {
		log.Printf("%s", mustFormatNode(token.NewFileSet(), blockcopy))
		log.Fatal("internal error: parent doesn't end in a return")
	}
	return blockcopy
}

// in the case that there's a return in the body being inlined (toBlock), addContinues
// replaces those returns that are not at the end of the function with the code in the
// caller after the function call that execution would continue with after the return.
// The block being added must end in a return.
func addContinues(cursor *astutil.Cursor, assignNode *ast.AssignStmt, toBlock *ast.BlockStmt, continueBlock *ast.BlockStmt) ast.Node {
	if !hasTerminatingReturn(continueBlock) {
		log.Fatal("the block being continued to in addContinues must end in a return")
	}
	applyFunc := func(cursor *astutil.Cursor) bool {
		ret, ok := cursor.Node().(*ast.ReturnStmt)
		if !ok {
			return true
		}

		if cursor.Parent() == toBlock && cursor.Index() == len(toBlock.List)-1 {
			return false
		}

		// This is the opposite of replacing a function call with the body. First
		// we replace the return statement with the assignment from the caller, and
		// then add the code we continue with.
		replaceWithAssignment(cursor, assignNode.Lhs, ret.Results, assignNode.Tok)
		cursor.InsertAfter(internalastutil.CloneNode(continueBlock))

		return false
	}
	return astutil.Apply(toBlock, applyFunc, nil)
}

// debugPrint prints out the expressions given by nodes for debugging.
func debugPrint(nodes ...ast.Expr) string {
	var b strings.Builder
	for i, node := range nodes {
		b.Write(mustFormatNode(token.NewFileSet(), node))
		if i != len(nodes)-1 {
			b.WriteString(", ")
		}
	}
	return b.String()
}

// mustFormatNode produces the formatted Go code for the given node.
func mustFormatNode(fset *token.FileSet, node any) []byte {
	var buf bytes.Buffer
	format.Node(&buf, fset, node)
	return buf.Bytes()
}

// mustMatchExprs makes sure that the expression lists have the same length,
// and returns the lists of the expressions on the lhs and rhs where the
// identifiers are not the same. These are used to produce assignment statements
// where the expressions on the right are assigned to the identifiers on the left.
func mustMatchExprs(lhs []ast.Expr, rhs []ast.Expr) ([]ast.Expr, []ast.Expr) {
	if len(lhs) != len(rhs) {
		log.Fatal("exprs don't match", debugPrint(lhs...), debugPrint(rhs...))
	}

	var newLhs, newRhs []ast.Expr
	for i := range lhs {
		lhsIdent, ok1 := lhs[i].(*ast.Ident)
		rhsIdent, ok2 := rhs[i].(*ast.Ident)
		if ok1 && ok2 && lhsIdent.Name == rhsIdent.Name {
			continue
		}
		newLhs = append(newLhs, lhs[i])
		newRhs = append(newRhs, rhs[i])
	}

	return newLhs, newRhs
}

// replaceWithAssignment replaces the node pointed to by the cursor with an assignment of the
// left hand side to the righthand side, removing any redundant assignments of a variable to itself,
// and replacing an assignment to a single basic literal with a constant declaration.
func replaceWithAssignment(cursor *astutil.Cursor, lhs, rhs []ast.Expr, tok token.Token) {
	newLhs, newRhs := mustMatchExprs(lhs, rhs)
	if len(newLhs) == 0 {
		cursor.Delete()
		return
	}
	if len(newRhs) == 1 {
		if lit, ok := newRhs[0].(*ast.BasicLit); ok {
			constDecl := &ast.DeclStmt{
				Decl: &ast.GenDecl{
					Tok: token.CONST,
					Specs: []ast.Spec{
						&ast.ValueSpec{
							Names:  []*ast.Ident{newLhs[0].(*ast.Ident)},
							Values: []ast.Expr{lit},
						},
					},
				},
			}
			cursor.Replace(constDecl)
			return
		}
	}
	newAssignment := &ast.AssignStmt{
		Lhs: newLhs,
		Rhs: newRhs,
		Tok: tok,
	}
	cursor.Replace(newAssignment)
}

// generateTable generates the file with the jump tables for the specialized malloc functions.
func generateTable(sizeToSizeClass []uint8) []byte {
	scMax := sizeToSizeClass[specializedMallocMax]

	var b bytes.Buffer
	fmt.Fprintf(&b, `// Code generated by mkmalloc.go; DO NOT EDIT.
//go:build !plan9

package runtime

import "unsafe"

var mallocScanTable = [%d]func(size uintptr, typ *_type, needzero bool) unsafe.Pointer{`, specializedMallocMax+1)

	for i := range uintptr(specializedMallocMax + 1) {
		fmt.Fprintf(&b, "%s,\n", smallScanNoHeaderSCFuncName(sizeToSizeClass[i], scMax))
	}

	fmt.Fprintf(&b, `
}

var mallocNoScanTable = [%d]func(size uintptr, typ *_type, needzero bool) unsafe.Pointer{`, specializedMallocMax+1)
	for i := range uintptr(specializedMallocMax + 1) {
		if i < 16 {
			fmt.Fprintf(&b, "%s,\n", "mallocPanic")
		} else {
			fmt.Fprintf(&b, "%s,\n", smallNoScanSCFuncName(sizeToSizeClass[i], scMax))
		}
	}

	fmt.Fprintln(&b, `
}`)

	return b.Bytes()
}

// Generate benchmarks for all potentially small sizes
// (sizes for which smallScanNoHeader would be called)
// gc.MinSizeForMallocHeader is defined as goarch.PtrSize * goarch.PtrBits.

const benchmarkMax = maxPtrSize * maxPtrBits

// benchmarkConfig produces an inlining config to stamp out microbenchmarks.
func benchmarkConfig(classes []class, sizeToSizeClass []uint8) generatorConfig {
	config := generatorConfig{file: "../malloc_stubs_test.go"}

	scMax := sizeToSizeClass[benchmarkMax]

	str := fmt.Sprint

	for sc := uint8(1); sc <= scMax; sc++ {
		elemsize := classes[sc].size
		config.specs = append(config.specs, spec{
			templateFunc: "benchmarkStub",
			name:         fmt.Sprintf("benchmarkMallocgcNoscan%d", elemsize),
			ops: []op{
				{subBasicLit, "size_", str(elemsize)},
				{foldCondition, "noscan_", str(true)},
			},
		})
		config.specs = append(config.specs, spec{
			templateFunc: "benchmarkStub",
			name:         fmt.Sprintf("benchmarkMallocgcScan%d", elemsize),
			ops: []op{
				{subBasicLit, "size_", str(elemsize)},
				{foldCondition, "noscan_", str(false)},
			},
		})
		config.specs = append(config.specs, spec{
			templateFunc: "benchmarkScanSliceStub",
			name:         fmt.Sprintf("benchmarkMallocgcScanSlice%d", elemsize),
			ops:          []op{{subBasicLit, "size_", str(elemsize)}},
		})
	}

	for size := 1; size < tinySize; size++ {
		config.specs = append(config.specs, spec{
			templateFunc: "benchmarkStubTiny",
			name:         fmt.Sprintf("benchmarkMallocgcTiny%d", size),
			ops:          []op{{subBasicLit, "size_", str(size)}, {foldCondition, "noscan_", str(true)}},
		})
	}

	return config
}

func generateTopBenchmark(classes []class, sizeToSizeClass []uint8) string {
	scMax := sizeToSizeClass[benchmarkMax]
	bench := `func BenchmarkMallocgc(b *testing.B) {
		b.Run("scan=noscan", func(b *testing.B) {
`
	for size := 1; size < tinySize; size++ {
		bench += fmt.Sprintf(`b.Run("size=%d", benchmarkMallocgcTiny%d)`, size, size) + "\n"
	}
	for sc := uint8(2); sc <= scMax; sc++ {
		elemsize := classes[sc].size
		bench += fmt.Sprintf(`b.Run("size=%d", benchmarkMallocgcNoscan%d)`, elemsize, elemsize) + "\n"
	}
	bench += `})
		b.Run("scan=scan", func(b *testing.B) {
`
	for sc := uint8(1); sc <= scMax; sc++ {
		elemsize := classes[sc].size
		bench += fmt.Sprintf(`b.Run("size=%d", benchmarkMallocgcScan%d)`, elemsize, elemsize) + "\n"

	}
	bench += `})
		b.Run("scan=scanslice", func(b *testing.B) {
`
	for sc := uint8(1); sc <= scMax; sc++ {
		elemsize := classes[sc].size
		bench += fmt.Sprintf(`b.Run("size=%d", benchmarkMallocgcScanSlice%d)`, elemsize, elemsize) + "\n"
	}
	bench += `})
}`

	return bench
}
