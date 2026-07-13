// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

import (
	"fmt"
	"go/types"
	"regexp"
	"simd/archsimd/_gen/specgen/specexpr"
	"strings"
)

func (sFn *specFunc) expand(ctx context, opts *LoadOptions) []*Func {
	ctx = ctx.at(sFn.Pos)

	var solver specexpr.Solver

	if opts.Trace != nil {
		fmt.Fprintf(opts.Trace, "## %s%s\n", sFn.Name, sFn.Sig)
		solver.SetTrace(opts.Trace)
	}

	// Declare domains of type parameters
	typeParamVars := make(map[*types.TypeParam]specexpr.Variable)
	for _, param := range sFn.TypeParams {
		types, err := constraintToDomain(sFn.Pkg, param.Constraint())
		if err != nil {
			panic(err)
		}
		varDef := specexpr.Variable("$" + param.String())
		solver.Declare(varDef, types)
		typeParamVars[param] = varDef
	}
	// Bind shapes of all function parameters and results of Vec type
	b := &argBinder{ctx, sFn.Pkg, &solver, typeParamVars}
	argGet := make(map[*types.Var]func(*specexpr.Bindings) specexpr.Type)
	ok := true
	for _, v := range sFn.Params {
		get := b.bindArg(v.Name(), v.Type())
		ok = ok && (get != nil)
		argGet[v] = get
	}
	for _, v := range sFn.Results {
		get := b.bindArg(v.Name(), v.Type())
		ok = ok && (get != nil)
		argGet[v] = get
	}
	// Add requirements to the solver
	for _, expr := range sFn.Requirements {
		solver.Assert(expr)
	}
	if !ok {
		return nil
	}

	// Find solutions.
	defer func() {
		p := recover()
		if p != nil {
			var buf strings.Builder
			solver.Fprint(&buf)
			panic(fmt.Sprintf("%s: %s\n%s", ctx.root.fset.Position(sFn.Pos), p, buf.String()))
		}
	}()
	var funcs []*Func
	for soln, err := range solver.Solve() {
		if err != nil {
			ctx.errorf("%s", err)
			continue
		}

		fn := sFn.instantiate(ctx, soln, argGet)
		if fn == nil {
			continue
		}
		fn.typeParamVars = typeParamVars

		funcs = append(funcs, fn)
	}

	if len(funcs) == 0 {
		ctx.errorf("impossible constraints (try -f %s -trace)", sFn.Name)
	}

	return funcs
}

func (sFn *specFunc) instantiate(ctx context, b *specexpr.Bindings, argGet map[*types.Var]func(*specexpr.Bindings) specexpr.Type) *Func {
	var f Func

	f.specFunc = sFn
	f.instance = b

	// Function or method?
	var method bool
	if len(sFn.Params) > 0 {
		if t, ok := sFn.Params[0].Type().(*types.Named); ok {
			if t.Origin() == sFn.Pkg.VecType {
				method = true
			}
		}
	}

	// Instantiate name
	name := sFn.NameTmpl.expand(func(s string) string {
		val := b.Get(specexpr.Variable(s))
		if val == nil {
			ctx.errorf("unknown variable %q in function name", s)
			return ""
		}
		str := fmt.Sprint(val)
		// Make sure str starts with an upper-case letter so it maintains
		// CamelCase in the overall identifier.
		str = strings.ToTitle(str[:1]) + str[1:]
		return str
	})
	f.Name = name

	// Instantiate doc
	doc := sFn.Doc.expand(func(s string) string {
		val := b.Get(specexpr.Variable(s))
		if val == nil {
			ctx.errorf("unknown variable %q in doc", s)
			return ""
		}
		return fmt.Sprint(val)
	})
	// Replace name in doc
	if f.Name == sFn.Name {
		f.Doc = doc
	} else {
		f.Doc = regexp.MustCompile(`\b`+regexp.QuoteMeta(sFn.Name)+`\b`).ReplaceAllLiteralString(doc, f.Name)
	}

	// Instantiate parameter and result types
	//
	// TODO: Should the loader keep these grouped like the original source so
	// the transformed version keeps the same grouping (modulo pulling off the
	// receiver)?
	for _, v := range sFn.Params {
		t := argGet[v](b)
		f.In = append(f.In, Arg{v.Name(), t})
	}
	if method && len(f.In) > 0 {
		f.Recv = f.In[0]
		f.In = f.In[1:]
	}
	for _, v := range sFn.Results {
		t := argGet[v](b)
		f.Out = append(f.Out, Arg{v.Name(), t})
	}

	return &f
}
