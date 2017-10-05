// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/ast"
	"go/token"
	"os"
	"reflect"
	"strings"
)

// Partial type checker.
//
// The fact that it is partial is very important: the input is
// an AST and a description of some type information to
// assume about one or more packages, but not all the
// packages that the program imports. The checker is
// expected to do as much as it can with what it has been
// given. There is not enough information supplied to do
// a full type check, but the type checker is expected to
// apply information that can be derived from variable
// declarations, function and method returns, and type switches
// as far as it can, so that the caller can still tell the types
// of expression relevant to a particular fix.
//
// TODO(rsc,gri): Replace with go/typechecker.
// Doing that could be an interesting test case for go/typechecker:
// the constraints about working with partial information will
// likely exercise it in interesting ways. The ideal interface would
// be to pass typecheck a map from importpath to package API text
// (Go source code), but for now we use data structures (TypeConfig, Type).
//
// The strings mostly use gofmt form.
//
// A Field or FieldList has as its type a comma-separated list
// of the types of the fields. For example, the field list
//	x, y, z int
// has type "int, int, int".

// The prefix "type " is the type of a type.
// For example, given
//	var x int
//	type T int
// x's type is "int" but T's type is "type int".
// mkType inserts the "type " prefix.
// getType removes it.
// isType tests for it.

func mkType(t string) string {
	return "type " + t
}

func getType(t string) string {
	if !isType(t) {
		return ""
	}
	return t[len("type "):]
}

func isType(t string) bool {
	return strings.HasPrefix(t, "type ")
}

// TypeConfig describes the universe of relevant types.
// For ease of creation, the types are all referred to by string
// name (e.g., "reflect.Value").  TypeByName is the only place
// where the strings are resolved.

type TypeConfig struct {
	Type map[string]*Type
	Var  map[string]string
	Func map[string]string
}

// typeof returns the type of the given name, which may be of
// the form "x" or "p.X".
func (cfg *TypeConfig) typeof(name string) string {
	if cfg.Var != nil {
		if t := cfg.Var[name]; t != "" {
			return t
		}
	}
	if cfg.Func != nil {
		if t := cfg.Func[name]; t != "" {
			return "func()" + t
		}
	}
	return ""
}

// Type describes the Fields and Methods of a type.
// If the field or method cannot be found there, it is next
// looked for in the Embed list.
type Type struct {
	Field  map[string]string // map field name to type
	Method map[string]string // map method name to comma-separated return types (should start with "func ")
	Embed  []string          // list of types this type embeds (for extra methods)
	Def    string            // definition of named type
}

// dot returns the type of "typ.name", making its decision
// using the type information in cfg.
func (typ *Type) dot(cfg *TypeConfig, name string) string {
	if typ.Field != nil {
		if t := typ.Field[name]; t != "" {
			return t
		}
	}
	if typ.Method != nil {
		if t := typ.Method[name]; t != "" {
			return t
		}
	}

	for _, e := range typ.Embed {
		etyp := cfg.Type[e]
		if etyp != nil {
			if t := etyp.dot(cfg, name); t != "" {
				return t
			}
		}
	}

	return ""
}

// typecheck type checks the AST f assuming the information in cfg.
// It returns two maps with type information:
// typeof maps AST nodes to type information in gofmt string form.
// assign maps type strings to lists of expressions that were assigned
// to values of another type that were assigned to that type.
func typecheck(cfg *TypeConfig, f *ast.File) (typeof map[interface{}]string, assign map[string][]interface{}) {
	typeof = make(map[interface{}]string)
	assign = make(map[string][]interface{})
	cfg1 := &TypeConfig{}
	*cfg1 = *cfg // make copy so we can add locally
	copied := false

	// gather function declarations
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		typecheck1(cfg, fn.Type, typeof, assign)
		t := typeof[fn.Type]
		if fn.Recv != nil {
			// The receiver must be a type.
			rcvr := typeof[fn.Recv]
			if !isType(rcvr) {
				if len(fn.Recv.List) != 1 {
					continue
				}
				rcvr = mkType(gofmt(fn.Recv.List[0].Type))
				typeof[fn.Recv.List[0].Type] = rcvr
			}
			rcvr = getType(rcvr)
			if rcvr != "" && rcvr[0] == '*' {
				rcvr = rcvr[1:]
			}
			typeof[rcvr+"."+fn.Name.Name] = t
		} else {
			if isType(t) {
				t = getType(t)
			} else {
				t = gofmt(fn.Type)
			}
			typeof[fn.Name] = t

			// Record typeof[fn.Name.Obj] for future references to fn.Name.
			typeof[fn.Name.Obj] = t
		}
	}

	// gather struct declarations
	for _, decl := range f.Decls {
		d, ok := decl.(*ast.GenDecl)
		if ok {
			for _, s := range d.Specs {
				switch s := s.(type) {
				case *ast.TypeSpec:
					if cfg1.Type[s.Name.Name] != nil {
						break
					}
					if !copied {
						copied = true
						// Copy map lazily: it's time.
						cfg1.Type = make(map[string]*Type)
						for k, v := range cfg.Type {
							cfg1.Type[k] = v
						}
					}
					t := &Type{Field: map[string]string{}}
					cfg1.Type[s.Name.Name] = t
					switch st := s.Type.(type) {
					case *ast.StructType:
						for _, f := range st.Fields.List {
							for _, n := range f.Names {
								t.Field[n.Name] = gofmt(f.Type)
							}
						}
					case *ast.ArrayType, *ast.StarExpr, *ast.MapType:
						t.Def = gofmt(st)
					}
				}
			}
		}
	}

	typecheck1(cfg1, f, typeof, assign)
	return typeof, assign
}

func makeExprList(a []*ast.Ident) []ast.Expr {
	var b []ast.Expr
	for _, x := range a {
		b = append(b, x)
	}
	return b
}

// Typecheck1 is the recursive form of typecheck.
// It is like typecheck but adds to the information in typeof
// instead of allocating a new map.
func typecheck1(cfg *TypeConfig, f interface{}, typeof map[interface{}]string, assign map[string][]interface{}) {
	// set sets the type of n to typ.
	// If isDecl is true, n is being declared.
	set := func(n ast.Expr, typ string, isDecl bool) {
		if typeof[n] != "" || typ == "" {
			if typeof[n] != typ {
				assign[typ] = append(assign[typ], n)
			}
			return
		}
		typeof[n] = typ

		// If we obtained typ from the declaration of x
		// propagate the type to all the uses.
		// The !isDecl case is a cheat here, but it makes
		// up in some cases for not paying attention to
		// struct fields. The real type checker will be
		// more accurate so we won't need the cheat.
		if id, ok := n.(*ast.Ident); ok && id.Obj != nil && (isDecl || typeof[id.Obj] == "") {
			typeof[id.Obj] = typ
		}
	}

	// Type-check an assignment lhs = rhs.
	// If isDecl is true, this is := so we can update
	// the types of the objects that lhs refers to.
	typecheckAssign := func(lhs, rhs []ast.Expr, isDecl bool) {
		if len(lhs) > 1 && len(rhs) == 1 {
			if _, ok := rhs[0].(*ast.CallExpr); ok {
				t := split(typeof[rhs[0]])
				// Lists should have same length but may not; pair what can be paired.
				for i := 0; i < len(lhs) && i < len(t); i++ {
					set(lhs[i], t[i], isDecl)
				}
				return
			}
		}
		if len(lhs) == 1 && len(rhs) == 2 {
			// x = y, ok
			rhs = rhs[:1]
		} else if len(lhs) == 2 && len(rhs) == 1 {
			// x, ok = y
			lhs = lhs[:1]
		}

		// Match as much as we can.
		for i := 0; i < len(lhs) && i < len(rhs); i++ {
			x, y := lhs[i], rhs[i]
			if typeof[y] != "" {
				set(x, typeof[y], isDecl)
			} else {
				set(y, typeof[x], false)
			}
		}
	}

	expand := func(s string) string {
		typ := cfg.Type[s]
		if typ != nil && typ.Def != "" {
			return typ.Def
		}
		return s
	}

	// The main type check is a recursive algorithm implemented
	// by walkBeforeAfter(n, before, after).
	// Most of it is bottom-up, but in a few places we need
	// to know the type of the function we are checking.
	// The before function records that information on
	// the curfn stack.
	var curfn []*ast.FuncType

	before := func(n interface{}) {
		// push function type on stack
		switch n := n.(type) {
		case *ast.FuncDecl:
			curfn = append(curfn, n.Type)
		case *ast.FuncLit:
			curfn = append(curfn, n.Type)
		}
	}

	// After is the real type checker.
	after := func(n interface{}) {
		if n == nil {
			return
		}
		if false && reflect.TypeOf(n).Kind() == reflect.Ptr { // debugging trace
			defer func() {
				if t := typeof[n]; t != "" {
					pos := fset.Position(n.(ast.Node).Pos())
					fmt.Fprintf(os.Stderr, "%s: typeof[%s] = %s\n", pos, gofmt(n), t)
				}
			}()
		}

		switch n := n.(type) {
		case *ast.FuncDecl, *ast.FuncLit:
			// pop function type off stack
			curfn = curfn[:len(curfn)-1]

		case *ast.FuncType:
			typeof[n] = mkType(joinFunc(split(typeof[n.Params]), split(typeof[n.Results])))

		case *ast.FieldList:
			// Field list is concatenation of sub-lists.
			t := ""
			for _, field := range n.List {
				if t != "" {
					t += ", "
				}
				t += typeof[field]
			}
			typeof[n] = t

		case *ast.Field:
			// Field is one instance of the type per name.
			all := ""
			t := typeof[n.Type]
			if !isType(t) {
				// Create a type, because it is typically *T or *p.T
				// and we might care about that type.
				t = mkType(gofmt(n.Type))
				typeof[n.Type] = t
			}
			t = getType(t)
			if len(n.Names) == 0 {
				all = t
			} else {
				for _, id := range n.Names {
					if all != "" {
						all += ", "
					}
					all += t
					typeof[id.Obj] = t
					typeof[id] = t
				}
			}
			typeof[n] = all

		case *ast.ValueSpec:
			// var declaration. Use type if present.
			if n.Type != nil {
				t := typeof[n.Type]
				if !isType(t) {
					t = mkType(gofmt(n.Type))
					typeof[n.Type] = t
				}
				t = getType(t)
				for _, id := range n.Names {
					set(id, t, true)
				}
			}
			// Now treat same as assignment.
			typecheckAssign(makeExprList(n.Names), n.Values, true)

		case *ast.AssignStmt:
			typecheckAssign(n.Lhs, n.Rhs, n.Tok == token.DEFINE)

		case *ast.Ident:
			// Identifier can take its type from underlying object.
			if t := typeof[n.Obj]; t != "" {
				typeof[n] = t
			}

		case *ast.SelectorExpr:
			// Field or method.
			name := n.Sel.Name
			if t := typeof[n.X]; t != "" {
				t = strings.TrimPrefix(t, "*") // implicit *
				if typ := cfg.Type[t]; typ != nil {
					if t := typ.dot(cfg, name); t != "" {
						typeof[n] = t
						return
					}
				}
				tt := typeof[t+"."+name]
				if isType(tt) {
					typeof[n] = getType(tt)
					return
				}
			}
			// Package selector.
			if x, ok := n.X.(*ast.Ident); ok && x.Obj == nil {
				str := x.Name + "." + name
				if cfg.Type[str] != nil {
					typeof[n] = mkType(str)
					return
				}
				if t := cfg.typeof(x.Name + "." + name); t != "" {
					typeof[n] = t
					return
				}
			}

		case *ast.CallExpr:
			// make(T) has type T.
			if isTopName(n.Fun, "make") && len(n.Args) >= 1 {
				typeof[n] = gofmt(n.Args[0])
				return
			}
			// new(T) has type *T
			if isTopName(n.Fun, "new") && len(n.Args) == 1 {
				typeof[n] = "*" + gofmt(n.Args[0])
				return
			}
			// Otherwise, use type of function to determine arguments.
			t := typeof[n.Fun]
			in, out := splitFunc(t)
			if in == nil && out == nil {
				return
			}
			typeof[n] = join(out)
			for i, arg := range n.Args {
				if i >= len(in) {
					break
				}
				if typeof[arg] == "" {
					typeof[arg] = in[i]
				}
			}

		case *ast.TypeAssertExpr:
			// x.(type) has type of x.
			if n.Type == nil {
				typeof[n] = typeof[n.X]
				return
			}
			// x.(T) has type T.
			if t := typeof[n.Type]; isType(t) {
				typeof[n] = getType(t)
			} else {
				typeof[n] = gofmt(n.Type)
			}

		case *ast.SliceExpr:
			// x[i:j] has type of x.
			typeof[n] = typeof[n.X]

		case *ast.IndexExpr:
			// x[i] has key type of x's type.
			t := expand(typeof[n.X])
			if strings.HasPrefix(t, "[") || strings.HasPrefix(t, "map[") {
				// Lazy: assume there are no nested [] in the array
				// length or map key type.
				if i := strings.Index(t, "]"); i >= 0 {
					typeof[n] = t[i+1:]
				}
			}

		case *ast.StarExpr:
			// *x for x of type *T has type T when x is an expr.
			// We don't use the result when *x is a type, but
			// compute it anyway.
			t := expand(typeof[n.X])
			if isType(t) {
				typeof[n] = "type *" + getType(t)
			} else if strings.HasPrefix(t, "*") {
				typeof[n] = t[len("*"):]
			}

		case *ast.UnaryExpr:
			// &x for x of type T has type *T.
			t := typeof[n.X]
			if t != "" && n.Op == token.AND {
				typeof[n] = "*" + t
			}

		case *ast.CompositeLit:
			// T{...} has type T.
			typeof[n] = gofmt(n.Type)

		case *ast.ParenExpr:
			// (x) has type of x.
			typeof[n] = typeof[n.X]

		case *ast.RangeStmt:
			t := expand(typeof[n.X])
			if t == "" {
				return
			}
			var key, value string
			if t == "string" {
				key, value = "int", "rune"
			} else if strings.HasPrefix(t, "[") {
				key = "int"
				if i := strings.Index(t, "]"); i >= 0 {
					value = t[i+1:]
				}
			} else if strings.HasPrefix(t, "map[") {
				if i := strings.Index(t, "]"); i >= 0 {
					key, value = t[4:i], t[i+1:]
				}
			}
			changed := false
			if n.Key != nil && key != "" {
				changed = true
				set(n.Key, key, n.Tok == token.DEFINE)
			}
			if n.Value != nil && value != "" {
				changed = true
				set(n.Value, value, n.Tok == token.DEFINE)
			}
			// Ugly failure of vision: already type-checked body.
			// Do it again now that we have that type info.
			if changed {
				typecheck1(cfg, n.Body, typeof, assign)
			}

		case *ast.TypeSwitchStmt:
			// Type of variable changes for each case in type switch,
			// but go/parser generates just one variable.
			// Repeat type check for each case with more precise
			// type information.
			as, ok := n.Assign.(*ast.AssignStmt)
			if !ok {
				return
			}
			varx, ok := as.Lhs[0].(*ast.Ident)
			if !ok {
				return
			}
			t := typeof[varx]
			for _, cas := range n.Body.List {
				cas := cas.(*ast.CaseClause)
				if len(cas.List) == 1 {
					// Variable has specific type only when there is
					// exactly one type in the case list.
					if tt := typeof[cas.List[0]]; isType(tt) {
						tt = getType(tt)
						typeof[varx] = tt
						typeof[varx.Obj] = tt
						typecheck1(cfg, cas.Body, typeof, assign)
					}
				}
			}
			// Restore t.
			typeof[varx] = t
			typeof[varx.Obj] = t

		case *ast.ReturnStmt:
			if len(curfn) == 0 {
				// Probably can't happen.
				return
			}
			f := curfn[len(curfn)-1]
			res := n.Results
			if f.Results != nil {
				t := split(typeof[f.Results])
				for i := 0; i < len(res) && i < len(t); i++ {
					set(res[i], t[i], false)
				}
			}
		}
	}
	walkBeforeAfter(f, before, after)
}

// Convert between function type strings and lists of types.
// Using strings makes this a little harder, but it makes
// a lot of the rest of the code easier. This will all go away
// when we can use go/typechecker directly.

// splitFunc splits "func(x,y,z) (a,b,c)" into ["x", "y", "z"] and ["a", "b", "c"].
func splitFunc(s string) (in, out []string) {
	if !strings.HasPrefix(s, "func(") {
		return nil, nil
	}

	i := len("func(") // index of beginning of 'in' arguments
	nparen := 0
	for j := i; j < len(s); j++ {
		switch s[j] {
		case '(':
			nparen++
		case ')':
			nparen--
			if nparen < 0 {
				// found end of parameter list
				out := strings.TrimSpace(s[j+1:])
				if len(out) >= 2 && out[0] == '(' && out[len(out)-1] == ')' {
					out = out[1 : len(out)-1]
				}
				return split(s[i:j]), split(out)
			}
		}
	}
	return nil, nil
}

// joinFunc is the inverse of splitFunc.
func joinFunc(in, out []string) string {
	outs := ""
	if len(out) == 1 {
		outs = " " + out[0]
	} else if len(out) > 1 {
		outs = " (" + join(out) + ")"
	}
	return "func(" + join(in) + ")" + outs
}

// split splits "int, float" into ["int", "float"] and splits "" into [].
func split(s string) []string {
	out := []string{}
	i := 0 // current type being scanned is s[i:j].
	nparen := 0
	for j := 0; j < len(s); j++ {
		switch s[j] {
		case ' ':
			if i == j {
				i++
			}
		case '(':
			nparen++
		case ')':
			nparen--
			if nparen < 0 {
				// probably can't happen
				return nil
			}
		case ',':
			if nparen == 0 {
				if i < j {
					out = append(out, s[i:j])
				}
				i = j + 1
			}
		}
	}
	if nparen != 0 {
		// probably can't happen
		return nil
	}
	if i < len(s) {
		out = append(out, s[i:])
	}
	return out
}

// join is the inverse of split.
func join(x []string) string {
	return strings.Join(x, ", ")
}
