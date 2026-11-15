// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func lowerFileBytes(t *testing.T, name, data string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(data), 0666); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var perrs []error
	f, err := syntax.ParseFile(path, func(err error) { perrs = append(perrs, err) }, nil, syntax.CheckBranches|syntax.LogosMode)
	if err != nil {
		t.Fatal(errors.Join(append(perrs, err)...))
	}
	imp := newTestImporter(t)
	info := &types2.Info{
		Defs:       make(map[*syntax.Name]types2.Object),
		Uses:       make(map[*syntax.Name]types2.Object),
		Types:      make(map[syntax.Expr]types2.TypeAndValue),
		Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
		Scopes:     make(map[syntax.Node]*types2.Scope),
	}
	var cerrs []error
	conf := types2.Config{
		Importer: imp,
		Error:    func(err error) { cerrs = append(cerrs, err) },
		Logos:    true,
	}
	files := []*syntax.File{f}
	pkg, _ := conf.Check("main", files, info)
	if err := errors.Join(cerrs...); err != nil {
		t.Fatalf("type check: %v", err)
	}
	_, edits := Lower(pkg, info, files, map[*syntax.File][]byte{f: raw})
	out, err := ApplyEdits(raw, f.Pos().FileBase(), edits[f])
	if err != nil {
		t.Fatal(err)
	}
	return string(out)
}

func TestTimeAfterStaysInLoopBody(t *testing.T) {
	data, err := os.ReadFile("../../../../../test/logos/run/time_after_no_hoist.lgo")
	if err != nil {
		t.Fatal(err)
	}
	out := lowerFileBytes(t, "timeafternohoist.lgo", string(data))
	if n := strings.Count(out, "time.After("); n != 1 {
		t.Errorf("lowered loop: got %d time.After( calls, want 1:\n%s", n, out)
	}
	forIdx := strings.Index(out, "for ")
	afterIdx := strings.Index(out, "time.After(")
	if forIdx < 0 || afterIdx < 0 || afterIdx < forIdx {
		t.Errorf("lowered loop: time.After at %d must stay after for at %d:\n%s", afterIdx, forIdx, out)
	}
	if got, want := strings.Count(out, "<-"), strings.Count(string(data), "<-"); got != want {
		t.Errorf("lowered loop: got %d channel operations, want %d (no extra drain read may appear):\n%s", got, want, out)
	}
}
