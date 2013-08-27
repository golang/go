package oracle

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
	"sort"
	"strconv"
	"strings"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/importer"
	"code.google.com/p/go.tools/pointer"
	"code.google.com/p/go.tools/ssa"
)

// TODO(adonovan): all printed sets must be sorted to ensure test determinism.

// describe describes the syntax node denoted by the query position,
// including:
// - its syntactic category
// - the location of the definition of its referent (for identifiers)
// - its type and method set (for an expression or type expression)
// - its points-to set (for a pointer-like expression)
// - its concrete types (for an interface expression) and their points-to sets.
//
func describe(o *oracle) (queryResult, error) {
	if false { // debugging
		o.printf(o.queryPath[0], "you selected: %s %s",
			importer.NodeDescription(o.queryPath[0]), pathToString2(o.queryPath))
	}

	path, action := findInterestingNode(o.queryPkgInfo, o.queryPath)
	switch action {
	case actionExpr:
		return describeValue(o, path)

	case actionType:
		return describeType(o, path)

	case actionPackage:
		return describePackage(o, path)

	case actionStmt:
		return describeStmt(o, path)

	case actionUnknown:
		return &describeUnknownResult{path[0]}, nil

	default:
		panic(action) // unreachable
	}
}

type describeUnknownResult struct {
	node ast.Node
}

func (r *describeUnknownResult) display(o *oracle) {
	// Nothing much to say about misc syntax.
	o.printf(r.node, "%s", importer.NodeDescription(r.node))
}

type action int

const (
	actionUnknown action = iota // None of the below
	actionExpr                  // FuncDecl, true Expr or Ident(types.{Const,Var})
	actionType                  // type Expr or Ident(types.TypeName).
	actionStmt                  // Stmt or Ident(types.Label)
	actionPackage               // Ident(types.Package) or ImportSpec
)

// findInterestingNode classifies the syntax node denoted by path as one of:
//    - an expression, part of an expression or a reference to a constant
//      or variable;
//    - a type, part of a type, or a reference to a named type;
//    - a statement, part of a statement, or a label referring to a statement;
//    - part of a package declaration or import spec.
//    - none of the above.
// and returns the most "interesting" associated node, which may be
// the same node, an ancestor or a descendent.
//
func findInterestingNode(pkginfo *importer.PackageInfo, path []ast.Node) ([]ast.Node, action) {
	// TODO(adonovan): integrate with go/types/stdlib_test.go and
	// apply this to every AST node we can find to make sure it
	// doesn't crash.

	// TODO(adonovan): audit for ParenExpr safety, esp. since we
	// traverse up and down.

	// TODO(adonovan): if the users selects the "." in
	// "fmt.Fprintf()", they'll get an ambiguous selection error;
	// we won't even reach here.  Can we do better?

	// TODO(adonovan): describing a field within 'type T struct {...}'
	// describes the (anonymous) struct type and concludes "no methods".  Fix.

	for len(path) > 0 {
		switch n := path[0].(type) {
		case *ast.GenDecl:
			if len(n.Specs) == 1 {
				// Descend to sole {Import,Type,Value}Spec child.
				path = append([]ast.Node{n.Specs[0]}, path...)
				continue
			}
			return path, actionUnknown // uninteresting

		case *ast.FuncDecl:
			// Descend to function name.
			path = append([]ast.Node{n.Name}, path...)
			continue

		case *ast.ImportSpec:
			return path, actionPackage

		case *ast.ValueSpec:
			if len(n.Names) == 1 {
				// Descend to sole Ident child.
				path = append([]ast.Node{n.Names[0]}, path...)
				continue
			}
			return path, actionUnknown // uninteresting

		case *ast.TypeSpec:
			// Descend to type name.
			path = append([]ast.Node{n.Name}, path...)
			continue

		case ast.Stmt:
			return path, actionStmt

		case *ast.ArrayType,
			*ast.StructType,
			*ast.FuncType,
			*ast.InterfaceType,
			*ast.MapType,
			*ast.ChanType:
			return path, actionType

		case *ast.Comment, *ast.CommentGroup, *ast.File, *ast.KeyValueExpr, *ast.CommClause:
			return path, actionUnknown // uninteresting

		case *ast.Ellipsis:
			// Continue to enclosing node.
			// e.g. [...]T in ArrayType
			//      f(x...) in CallExpr
			//      f(x...T) in FuncType

		case *ast.Field:
			// TODO(adonovan): this needs more thought,
			// since fields can be so many things.
			if len(n.Names) == 1 {
				// Descend to sole Ident child.
				path = append([]ast.Node{n.Names[0]}, path...)
				continue
			}
			// Zero names (e.g. anon field in struct)
			// or multiple field or param names:
			// continue to enclosing field list.

		case *ast.FieldList:
			// Continue to enclosing node:
			// {Struct,Func,Interface}Type or FuncDecl.

		case *ast.BasicLit:
			if _, ok := path[1].(*ast.ImportSpec); ok {
				return path[1:], actionPackage
			}
			return path, actionExpr

		case *ast.SelectorExpr:
			if pkginfo.ObjectOf(n.Sel) == nil {
				// Is this reachable?
				return path, actionUnknown
			}
			// Descend to .Sel child.
			path = append([]ast.Node{n.Sel}, path...)
			continue

		case *ast.Ident:
			switch obj := pkginfo.ObjectOf(n).(type) {
			case *types.Package:
				return path, actionPackage

			case *types.Const:
				return path, actionExpr

			case *types.Label:
				return path, actionStmt

			case *types.TypeName:
				return path, actionType

			case *types.Var:
				// For x in 'struct {x T}', return struct type, for now.
				if _, ok := path[1].(*ast.Field); ok {
					_ = path[2].(*ast.FieldList) // assertion
					if _, ok := path[3].(*ast.StructType); ok {
						return path[3:], actionType
					}
				}
				return path, actionExpr

			case *types.Func:
				// For f in 'interface {f()}', return the interface type, for now.
				if _, ok := path[1].(*ast.Field); ok {
					_ = path[2].(*ast.FieldList) // assertion
					if _, ok := path[3].(*ast.InterfaceType); ok {
						return path[3:], actionType
					}
				}

				// For reference to built-in function, return enclosing call.
				if _, ok := obj.Type().(*types.Builtin); ok {
					// Ascend to enclosing function call.
					path = path[1:]
					continue
				}

				return path, actionExpr
			}

			// No object.
			switch path[1].(type) {
			case *ast.SelectorExpr:
				// Return enclosing selector expression.
				return path[1:], actionExpr

			case *ast.Field:
				// TODO(adonovan): test this.
				// e.g. all f in:
				//  struct { f, g int }
				//  interface { f() }
				//  func (f T) method(f, g int) (f, g bool)
				//
				// switch path[3].(type) {
				// case *ast.FuncDecl:
				// case *ast.StructType:
				// case *ast.InterfaceType:
				// }
				//
				// return path[1:], actionExpr
				//
				// Unclear what to do with these.
				// Struct.Fields             -- field
				// Interface.Methods         -- field
				// FuncType.{Params.Results} -- actionExpr
				// FuncDecl.Recv             -- actionExpr

			case *ast.ImportSpec:
				// TODO(adonovan): fix: why no package object? go/types bug?
				return path[1:], actionPackage

			default:
				// e.g. blank identifier (go/types bug?)
				// or y in "switch y := x.(type)" (go/types bug?)
				fmt.Printf("unknown reference %s in %T\n", n, path[1])
				return path, actionUnknown
			}

		case *ast.StarExpr:
			if pkginfo.IsType(n) {
				return path, actionType
			}
			return path, actionExpr

		case ast.Expr:
			// All Expr but {BasicLit,Ident,StarExpr} are
			// "true" expressions that evaluate to a value.
			return path, actionExpr
		}

		// Ascend to parent.
		path = path[1:]
	}

	return nil, actionUnknown // unreachable
}

// ---- VALUE ------------------------------------------------------------

// ssaValueForIdent returns the ssa.Value for the ast.Ident whose path
// to the root of the AST is path.  It may return a nil Value without
// an error to indicate the pointer analysis is not appropriate.
//
func ssaValueForIdent(o *oracle, obj types.Object, path []ast.Node) (ssa.Value, error) {
	if obj, ok := obj.(*types.Var); ok {
		pkg := o.prog.Package(o.queryPkgInfo.Pkg)
		pkg.Build()
		if v := o.prog.VarValue(obj, pkg, path); v != nil {
			// Don't run pointer analysis on a ref to a const expression.
			if _, ok := v.(*ssa.Const); ok {
				v = nil
			}
			return v, nil
		}
		return nil, fmt.Errorf("can't locate SSA Value for var %s", obj.Name())
	}

	// Don't run pointer analysis on const/func objects.
	return nil, nil
}

// ssaValueForExpr returns the ssa.Value of the non-ast.Ident
// expression whose path to the root of the AST is path.  It may
// return a nil Value without an error to indicate the pointer
// analysis is not appropriate.
//
func ssaValueForExpr(o *oracle, path []ast.Node) (ssa.Value, error) {
	pkg := o.prog.Package(o.queryPkgInfo.Pkg)
	pkg.SetDebugMode(true)
	pkg.Build()

	fn := ssa.EnclosingFunction(pkg, path)
	if fn == nil {
		return nil, fmt.Errorf("no SSA function built for this location (dead code?)")
	}

	if v := fn.ValueForExpr(path[0].(ast.Expr)); v != nil {
		return v, nil
	}

	return nil, fmt.Errorf("can't locate SSA Value for expression in %s", fn)
}

func describeValue(o *oracle, path []ast.Node) (*describeValueResult, error) {
	var expr ast.Expr
	switch n := path[0].(type) {
	case *ast.ValueSpec:
		// ambiguous ValueSpec containing multiple names
		return nil, o.errorf(n, "multiple value specification")
	case ast.Expr:
		expr = n
	default:
		// Is this reachable?
		return nil, o.errorf(n, "unexpected AST for expr: %T", n)
	}

	// From this point on, we cannot fail with an error.
	// Failure to run the pointer analysis will be reported later.

	var value ssa.Value
	var ptaErr error
	var obj types.Object

	// Determine the ssa.Value for the expression.
	if id, ok := expr.(*ast.Ident); ok {
		// def/ref of func/var/const object
		obj = o.queryPkgInfo.ObjectOf(id)
		value, ptaErr = ssaValueForIdent(o, obj, path)
	} else {
		// any other expression
		if o.queryPkgInfo.ValueOf(expr) == nil { // non-constant?
			value, ptaErr = ssaValueForExpr(o, path)
		}
	}

	// Don't run pointer analysis on non-pointerlike types.
	if value != nil && !pointer.CanPoint(value.Type()) {
		value = nil
	}

	// Run pointer analysis of the selected SSA value.
	var ptrs []pointer.Pointer
	if value != nil {
		buildSSA(o)

		o.config.QueryValues = map[ssa.Value][]pointer.Pointer{value: nil}
		ptrAnalysis(o)
		ptrs = o.config.QueryValues[value]
	}

	return &describeValueResult{
		expr:   expr,
		obj:    obj,
		value:  value,
		ptaErr: ptaErr,
		ptrs:   ptrs,
	}, nil
}

type describeValueResult struct {
	expr   ast.Expr          // query node
	obj    types.Object      // var/func/const object, if expr was Ident
	value  ssa.Value         // ssa.Value for pointer analysis query
	ptaErr error             // explanation of why we couldn't run pointer analysis
	ptrs   []pointer.Pointer // result of pointer analysis query
}

func (r *describeValueResult) display(o *oracle) {
	suffix := ""
	if val := o.queryPkgInfo.ValueOf(r.expr); val != nil {
		suffix = fmt.Sprintf(" of constant value %s", val)
	}

	// Describe the expression.
	if r.obj != nil {
		if r.obj.Pos() == r.expr.Pos() {
			// defining ident
			o.printf(r.expr, "definition of %s%s", r.obj, suffix)
		} else {
			// referring ident
			o.printf(r.expr, "reference to %s%s", r.obj, suffix)
			if def := r.obj.Pos(); def != token.NoPos {
				o.printf(def, "defined here")
			}
		}
	} else {
		desc := importer.NodeDescription(r.expr)
		if suffix != "" {
			// constant expression
			o.printf(r.expr, "%s%s", desc, suffix)
		} else {
			// non-constant expression
			o.printf(r.expr, "%s of type %s", desc, o.queryPkgInfo.TypeOf(r.expr))
		}
	}

	if r.value == nil {
		// pointer analysis was not run
		if r.ptaErr != nil {
			o.printf(r.expr, "no pointer analysis: %s", r.ptaErr)
		}
		return
	}

	if r.ptrs == nil {
		o.printf(r.expr, "pointer analysis did not analyze this expression (dead code?)")
		return
	}

	// Display the results of pointer analysis.

	// Combine the PT sets from all contexts.
	pts := pointer.PointsToCombined(r.ptrs)

	// Report which make(chan) labels the query's channel can alias.
	if _, ok := r.value.Type().Underlying().(*types.Interface); ok {
		// Show concrete types for interface expression.
		if concs := pts.ConcreteTypes(); concs.Len() > 0 {
			o.printf(o, "interface may contain these concrete types:")
			// TODO(adonovan): must sort to ensure deterministic test behaviour.
			concs.Iterate(func(conc types.Type, ptrs interface{}) {
				var obj types.Object
				if nt, ok := deref(conc).(*types.Named); ok {
					obj = nt.Obj()
				}

				pts := pointer.PointsToCombined(ptrs.([]pointer.Pointer))
				if labels := pts.Labels(); len(labels) > 0 {
					o.printf(obj, "\t%s, may point to:", conc)
					printLabels(o, labels, "\t\t")
				} else {
					o.printf(obj, "\t%s", conc)
				}
			})
		} else {
			o.printf(o, "interface cannot contain any concrete values.")
		}
	} else {
		// Show labels for other expressions.
		if labels := pts.Labels(); len(labels) > 0 {
			o.printf(o, "value may point to these labels:")
			printLabels(o, labels, "\t")
		} else {
			o.printf(o, "value cannot point to anything.")
		}
	}
}

type byPosAndString []*pointer.Label

func (a byPosAndString) Len() int { return len(a) }
func (a byPosAndString) Less(i, j int) bool {
	cmp := a[i].Pos() - a[j].Pos()
	return cmp < 0 || (cmp == 0 && a[i].String() < a[j].String())
}
func (a byPosAndString) Swap(i, j int) { a[i], a[j] = a[j], a[i] }

func printLabels(o *oracle, labels []*pointer.Label, prefix string) {
	// Sort, to ensure deterministic test behaviour.
	sort.Sort(byPosAndString(labels))
	// TODO(adonovan): due to context-sensitivity, many of these
	// labels may differ only by context, which isn't apparent.
	for _, label := range labels {
		o.printf(label, "%s%s", prefix, label)
	}
}

// ---- TYPE ------------------------------------------------------------

func describeType(o *oracle, path []ast.Node) (*describeTypeResult, error) {
	var description string
	var t types.Type
	switch n := path[0].(type) {
	case *ast.Ident:
		t = o.queryPkgInfo.TypeOf(n)
		switch t := t.(type) {
		case *types.Basic:
			description = "reference to built-in type " + t.String()

		case *types.Named:
			isDef := t.Obj().Pos() == n.Pos() // see caveats at isDef above
			if isDef {
				description = "definition of type " + t.String()
			} else {
				description = "reference to type " + t.String()
			}
		}

	case ast.Expr:
		t = o.queryPkgInfo.TypeOf(n)
		description = "type " + t.String()

	default:
		// Unreachable?
		return nil, o.errorf(n, "unexpected AST for type: %T", n)
	}

	return &describeTypeResult{path[0], description, t}, nil
}

type describeTypeResult struct {
	node        ast.Node
	description string
	typ         types.Type
}

func (r *describeTypeResult) display(o *oracle) {
	o.printf(r.node, "%s", r.description)

	// Show the underlying type for a reference to a named type.
	if nt, ok := r.typ.(*types.Named); ok && r.node.Pos() != nt.Obj().Pos() {
		o.printf(nt.Obj(), "defined as %s", nt.Underlying())
	}

	// Print the method set, if the type kind is capable of bearing methods.
	switch r.typ.(type) {
	case *types.Interface, *types.Struct, *types.Named:
		// TODO(adonovan): don't show unexported methods if
		// r.typ belongs to a package other than the query
		// package.
		if m := ssa.IntuitiveMethodSet(r.typ); m != nil {
			o.printf(r.node, "Method set:")
			for _, meth := range m {
				o.printf(meth.Obj(), "\t%s", meth)
			}
		} else {
			o.printf(r.node, "No methods.")
		}
	}
}

// ---- PACKAGE ------------------------------------------------------------

func describePackage(o *oracle, path []ast.Node) (*describePackageResult, error) {
	var description string
	var importPath string
	switch n := path[0].(type) {
	case *ast.ImportSpec:
		// importPath = o.queryPkgInfo.ObjectOf(n.Name).(*types.Package).Path()
		// description = "import of package " + importPath
		// TODO(gri): o.queryPkgInfo.ObjectOf(n.Name) may be nil.
		// e.g. "fmt" import in cmd/oracle/main.go.    Why?
		// Workaround:
		description = "import of package " + n.Path.Value
		importPath, _ = strconv.Unquote(n.Path.Value)

	case *ast.Ident:
		importPath = o.queryPkgInfo.ObjectOf(n).(*types.Package).Path()
		if _, isDef := path[1].(*ast.File); isDef {
			description = fmt.Sprintf("definition of package %q", importPath)
		} else {
			description = fmt.Sprintf("reference to package %q", importPath)
		}
		if importPath == "" {
			// TODO(gri): fix.
			return nil, o.errorf(n, "types.Package.Path() returned \"\"\n")
		}

	default:
		// Unreachable?
		return nil, o.errorf(n, "unexpected AST for package: %T", n)
	}

	pkg := o.prog.PackagesByPath[importPath]

	return &describePackageResult{path[0], description, pkg}, nil
}

type describePackageResult struct {
	node        ast.Node
	description string
	pkg         *ssa.Package
}

func (r *describePackageResult) display(o *oracle) {
	o.printf(r.node, "%s", r.description)
	// TODO(adonovan): factor this into a testable utility function.
	if p := r.pkg; p != nil {
		samePkg := p.Object == o.queryPkgInfo.Pkg

		// Describe exported package members, in lexicographic order.

		// Compute max width of name "column".
		var names []string
		maxname := 0
		for name := range p.Members {
			if samePkg || ast.IsExported(name) {
				if l := len(name); l > maxname {
					maxname = l
				}
				names = append(names, name)
			}
		}

		sort.Strings(names)

		// Print the members.
		for _, name := range names {
			mem := p.Members[name]
			o.printf(mem, "%s", formatMember(mem, maxname))
			// Print method set.
			if mem, ok := mem.(*ssa.Type); ok {
				for _, meth := range ssa.IntuitiveMethodSet(mem.Type()) {
					if samePkg || ast.IsExported(meth.Obj().Name()) {
						o.printf(meth.Obj(), "\t\t%s", meth)
					}
				}
			}
		}
	}
}

func formatMember(mem ssa.Member, maxname int) string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "\t%-5s %-*s", mem.Token(), maxname, mem.Name())
	switch mem := mem.(type) {
	case *ssa.NamedConst:
		fmt.Fprintf(&buf, " %s = %s", mem.Type(), mem.Value.Name())

	case *ssa.Function:
		fmt.Fprintf(&buf, " %s", mem.Type())

	case *ssa.Type:
		// Abbreviate long aggregate type names.
		var abbrev string
		switch t := mem.Type().Underlying().(type) {
		case *types.Interface:
			if t.NumMethods() > 1 {
				abbrev = "interface{...}"
			}
		case *types.Struct:
			if t.NumFields() > 1 {
				abbrev = "struct{...}"
			}
		}
		if abbrev == "" {
			fmt.Fprintf(&buf, " %s", mem.Type().Underlying())
		} else {
			fmt.Fprintf(&buf, " %s", abbrev)
		}

	case *ssa.Global:
		fmt.Fprintf(&buf, " %s", deref(mem.Type()))
	}
	return buf.String()
}

// ---- STATEMENT ------------------------------------------------------------

func describeStmt(o *oracle, path []ast.Node) (*describeStmtResult, error) {
	var description string
	switch n := path[0].(type) {
	case *ast.Ident:
		if o.queryPkgInfo.ObjectOf(n).Pos() == n.Pos() {
			description = "labelled statement"
		} else {
			description = "reference to labelled statement"
		}

	default:
		// Nothing much to say about statements.
		description = importer.NodeDescription(n)
	}
	return &describeStmtResult{path[0], description}, nil
}

type describeStmtResult struct {
	node        ast.Node
	description string
}

func (r *describeStmtResult) display(o *oracle) {
	o.printf(r.node, "%s", r.description)
}

// ------------------- Utilities -------------------

// pathToString returns a string containing the concrete types of the
// nodes in path.
func pathToString2(path []ast.Node) string {
	var buf bytes.Buffer
	fmt.Fprint(&buf, "[")
	for i, n := range path {
		if i > 0 {
			fmt.Fprint(&buf, " ")
		}
		fmt.Fprint(&buf, strings.TrimPrefix(fmt.Sprintf("%T", n), "*ast."))
	}
	fmt.Fprint(&buf, "]")
	return buf.String()
}
