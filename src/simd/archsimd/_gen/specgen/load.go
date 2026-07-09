// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

import (
	"cmp"
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"io"
	"maps"
	"simd/archsimd/_gen/specgen/specexpr"
	"slices"
	"strings"
)

type LoadOptions struct {
	// Filter, if non-nil, causes Load to process only spec functions satisfying
	// Filter.
	Filter func(*ast.FuncDecl) bool

	// Trace, if non-nil, causes Load to log a debug trace of solver steps to
	// Trace.
	Trace io.Writer
}

// Load loads a Go SIMD spec from the package in directory dir. This is the main
// entrypoint to this package.
func Load(dir string, opts *LoadOptions) ([]*Func, error) {
	if opts == nil {
		opts = new(LoadOptions)
	}

	var root contextRoot
	ctx := context{root: &root}

	pkg := loadSpecPackage(ctx, dir, opts)
	if err := root.gatherErrors(); err != nil {
		return nil, err
	}

	var allFuncs []*Func
	type funcKey struct {
		recv specexpr.Type
		name string
	}
	funcSet := make(map[funcKey]*Func)
	for _, sFn := range pkg.Funcs {
		expanded := sFn.expand(ctx, opts)

		// Check for duplicates
		for _, fn := range expanded {
			key := funcKey{fn.Recv.Type, fn.Name}
			if ofn := funcSet[key]; ofn != nil {
				ctx.at(sFn.Pos).errorf("conflicting functions:\n\t%s\n\t%s", fn.Signature(), ofn.Signature())
				continue
			}
			funcSet[key] = fn
			allFuncs = append(allFuncs, fn)
		}
	}

	return allFuncs, root.gatherErrors()
}

type contextRoot struct {
	fset   token.FileSet
	errors map[srcError]struct{}
}

type context struct {
	root *contextRoot
	pos  token.Pos
	fn   string
}

func (c context) at(pos token.Pos) context {
	c.pos = pos
	return c
}

func (c context) errorf(msg string, args ...any) {
	if c.root.errors == nil {
		c.root.errors = make(map[srcError]struct{})
	}
	err := srcError{c.pos, c.fn, fmt.Sprintf(msg, args...)}
	c.root.errors[err] = struct{}{}
}

func (r *contextRoot) gatherErrors() error {
	if len(r.errors) == 0 {
		return nil
	}
	var errs []error
	var buf strings.Builder
	for _, err := range slices.SortedFunc(maps.Keys(r.errors), func(a, b srcError) int {
		return cmp.Or(cmp.Compare(a.pos, b.pos), cmp.Compare(a.msg, b.msg))
	}) {
		if err.pos.IsValid() {
			fmt.Fprintf(&buf, "%s: ", r.fset.Position(err.pos))
		}
		buf.WriteString(err.msg)
		if err.fn != "" {
			fmt.Fprintf(&buf, " in %s", err.fn)
		}
		errs = append(errs, fmt.Errorf("%s", buf.String()))
		buf.Reset()
	}
	return errors.Join(errs...)
}

type srcError struct {
	pos token.Pos
	fn  string
	msg string
}
