// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

import (
	"fmt"
	"go/types"
	"simd/archsimd/_gen/specgen/specexpr"
	"strings"
)

// expand instantiates t over all solutions of its type parameters, producing
// one TypeDecl per API type.
func (t *specType) expand(ctx context, opts *LoadOptions) []*TypeDecl {
	ctx = ctx.at(t.Pos)

	var solver specexpr.Solver

	if opts.Trace != nil {
		fmt.Fprintf(opts.Trace, "## type %s\n", t.Name)
		solver.SetTrace(opts.Trace)
	}

	// Declare domains of type parameters
	typeParamVars := make(map[*types.TypeParam]specexpr.Variable)
	for _, param := range t.TypeParams {
		domain, err := constraintToDomain(t.Pkg, param.Constraint())
		if err != nil {
			panic(err)
		}
		varDef := specexpr.Variable("$" + param.String())
		solver.Declare(varDef, domain)
		typeParamVars[param] = varDef
	}

	// Bind the shape of the Vec this type is defined as
	b := &argBinder{ctx, t.Pkg, &solver, typeParamVars}
	get := b.bindArg("T", t.Def)
	if get == nil {
		return nil
	}

	defer func() {
		p := recover()
		if p != nil {
			var buf strings.Builder
			solver.Fprint(&buf)
			panic(fmt.Sprintf("%s: %s\n%s", ctx.root.fset.Position(t.Pos), p, buf.String()))
		}
	}()
	var decls []*TypeDecl
	for soln, err := range solver.Solve() {
		if err != nil {
			ctx.errorf("%s", err)
			continue
		}

		shape, ok := get(soln).(specexpr.Vector)
		if !ok {
			ctx.errorf("type %s does not describe a vector type", t.Name)
			continue
		}

		name, rawDoc := t.expandNameAndDoc(ctx, soln)
		if name != shape.String() {
			// API type names are determined by their shape, and consumers
			// look up types by name, so the name template must agree.
			ctx.errorf("type %s: name template produced %q for shape %s (use //specgen:name {{.T}})", t.Name, name, shape)
			continue
		}

		decls = append(decls, &TypeDecl{
			Name:  name,
			Shape: shape,
			Doc:   parseSpecDoc(ctx, rawDoc),
			Pos:   t.Pkg.Fset.Position(t.Pos),
		})
	}

	if len(decls) == 0 {
		ctx.errorf("impossible constraints for type %s", t.Name)
	}

	return decls
}
