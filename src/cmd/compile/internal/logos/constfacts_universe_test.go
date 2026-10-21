// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// TestEmitConstFactsUniverseMethod checks that a call of error.Error, a
// method of the predeclared error type that belongs to no package, is a
// fail-closed callee and not a nil dereference: slop::err_chain_reset code
// passes err.Error() as an argument.
func TestEmitConstFactsUniverseMethod(t *testing.T) {
	const src = `package factstest

func take(s string) {}

func Wrap(err error) {
	take(err.Error())
}
`
	path := filepath.Join(t.TempDir(), "universe.lgo")
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	file, err := syntax.ParseFile(path, func(err error) { t.Fatalf("parse: %v", err) }, nil, syntax.LogosMode)
	if err != nil {
		t.Fatal(err)
	}
	var checkErrs []error
	conf := types2.Config{
		Importer: newTestImporter(t),
		Error:    func(err error) { checkErrs = append(checkErrs, err) },
		Logos:    true,
	}
	info := &types2.Info{
		Types:      make(map[syntax.Expr]types2.TypeAndValue),
		Defs:       make(map[*syntax.Name]types2.Object),
		Uses:       make(map[*syntax.Name]types2.Object),
		Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
	}
	pkg, _ := conf.Check("factstest", []*syntax.File{file}, info)
	if err := errors.Join(checkErrs...); err != nil {
		t.Fatalf("check: %v", err)
	}
	EmitConstFacts(pkg, info, []*syntax.File{file}, nil)
}
