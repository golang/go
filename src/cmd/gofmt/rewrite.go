// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"reflect"
	"strings"
	"unicode"
	"utf8"
)


func initRewrite() {
	if *rewriteRule == "" {
		return
	}
	f := strings.Split(*rewriteRule, "->", 0)
	if len(f) != 2 {
		fmt.Fprintf(os.Stderr, "rewrite rule must be of the form 'pattern -> replacement'\n")
		os.Exit(2)
	}
	pattern := parseExpr(f[0], "pattern")
	replace := parseExpr(f[1], "replacement")
	rewrite = func(p *ast.File) *ast.File { return rewriteFile(pattern, replace, p) }
}


// parseExpr parses s as an expression.
// It might make sense to expand this to allow statement patterns,
// but there are problems with preserving formatting and also
// with what a wildcard for a statement looks like.
func parseExpr(s string, what string) ast.Expr {
	x, err := parser.ParseExpr("input", s)
	if err != nil {
		fmt.Fprintf(os.Stderr, "parsing %s %s: %s\n", what, s, err)
		os.Exit(2)
	}
	return x
}


// rewriteFile applys the rewrite rule pattern -> replace to an entire file.
func rewriteFile(pattern, replace ast.Expr, p *ast.File) *ast.File {
	m := make(map[string]reflect.Value)
	pat := reflect.NewValue(pattern)
	repl := reflect.NewValue(replace)
	var f func(val reflect.Value) reflect.Value // f is recursive
	f = func(val reflect.Value) reflect.Value {
		for k := range m {
			m[k] = nil, false
		}
		val = apply(f, val)
		if match(m, pat, val) {
			val = subst(m, repl, reflect.NewValue(val.Interface().(ast.Node).Pos()))
		}
		return val
	}
	return apply(f, reflect.NewValue(p)).Interface().(*ast.File)
}


var positionType = reflect.Typeof(token.Position{})
var identType = reflect.Typeof((*ast.Ident)(nil))


func isWildcard(s string) bool {
	rune, size := utf8.DecodeRuneInString(s)
	return size == len(s) && unicode.IsLower(rune)
}


// apply replaces each AST field x in val with f(x), returning val.
// To avoid extra conversions, f operates on the reflect.Value form.
func apply(f func(reflect.Value) reflect.Value, val reflect.Value) reflect.Value {
	if val == nil {
		return nil
	}
	switch v := reflect.Indirect(val).(type) {
	case *reflect.SliceValue:
		for i := 0; i < v.Len(); i++ {
			e := v.Elem(i)
			e.SetValue(f(e))
		}
	case *reflect.StructValue:
		for i := 0; i < v.NumField(); i++ {
			e := v.Field(i)
			e.SetValue(f(e))
		}
	case *reflect.InterfaceValue:
		e := v.Elem()
		v.SetValue(f(e))
	}
	return val
}


// match returns true if pattern matches val,
// recording wildcard submatches in m.
// If m == nil, match checks whether pattern == val.
func match(m map[string]reflect.Value, pattern, val reflect.Value) bool {
	// Wildcard matches any expression.  If it appears multiple
	// times in the pattern, it must match the same expression
	// each time.
	if m != nil && pattern.Type() == identType {
		name := pattern.Interface().(*ast.Ident).Value
		if isWildcard(name) {
			if old, ok := m[name]; ok {
				return match(nil, old, val)
			}
			m[name] = val
			return true
		}
	}

	// Otherwise, the expressions must match recursively.
	if pattern == nil || val == nil {
		return pattern == nil && val == nil
	}
	if pattern.Type() != val.Type() {
		return false
	}

	// Token positions need not match.
	if pattern.Type() == positionType {
		return true
	}

	p := reflect.Indirect(pattern)
	v := reflect.Indirect(val)

	switch p := p.(type) {
	case *reflect.SliceValue:
		v := v.(*reflect.SliceValue)
		if p.Len() != v.Len() {
			return false
		}
		for i := 0; i < p.Len(); i++ {
			if !match(m, p.Elem(i), v.Elem(i)) {
				return false
			}
		}
		return true

	case *reflect.StructValue:
		v := v.(*reflect.StructValue)
		if p.NumField() != v.NumField() {
			return false
		}
		for i := 0; i < p.NumField(); i++ {
			if !match(m, p.Field(i), v.Field(i)) {
				return false
			}
		}
		return true

	case *reflect.InterfaceValue:
		v := v.(*reflect.InterfaceValue)
		return match(m, p.Elem(), v.Elem())
	}

	// Handle token integers, etc.
	return p.Interface() == v.Interface()
}


// subst returns a copy of pattern with values from m substituted in place
// of wildcards and pos used as the position of tokens from the pattern.
// if m == nil, subst returns a copy of pattern and doesn't change the line
// number information.
func subst(m map[string]reflect.Value, pattern reflect.Value, pos reflect.Value) reflect.Value {
	if pattern == nil {
		return nil
	}

	// Wildcard gets replaced with map value.
	if m != nil && pattern.Type() == identType {
		name := pattern.Interface().(*ast.Ident).Value
		if isWildcard(name) {
			if old, ok := m[name]; ok {
				return subst(nil, old, nil)
			}
		}
	}

	if pos != nil && pattern.Type() == positionType {
		return pos
	}

	// Otherwise copy.
	switch p := pattern.(type) {
	case *reflect.SliceValue:
		v := reflect.MakeSlice(p.Type().(*reflect.SliceType), p.Len(), p.Len())
		for i := 0; i < p.Len(); i++ {
			v.Elem(i).SetValue(subst(m, p.Elem(i), pos))
		}
		return v

	case *reflect.StructValue:
		v := reflect.MakeZero(p.Type()).(*reflect.StructValue)
		for i := 0; i < p.NumField(); i++ {
			v.Field(i).SetValue(subst(m, p.Field(i), pos))
		}
		return v

	case *reflect.PtrValue:
		v := reflect.MakeZero(p.Type()).(*reflect.PtrValue)
		v.PointTo(subst(m, p.Elem(), pos))
		return v

	case *reflect.InterfaceValue:
		v := reflect.MakeZero(p.Type()).(*reflect.InterfaceValue)
		v.SetValue(subst(m, p.Elem(), pos))
		return v
	}

	return pattern
}
