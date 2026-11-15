// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cmd/compile/internal/logos/lint"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func TestEveryLintRuleHasFixture(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	lintDir := filepath.Join(testenv.GOROOT(t), "test", "logos", "lint")
	entries, err := os.ReadDir(lintDir)
	if err != nil {
		t.Fatal(err)
	}
	names := make(map[string]bool, len(entries))
	for _, e := range entries {
		names[e.Name()] = true
	}
	scriptDir := filepath.Join(testenv.GOROOT(t), "src", "cmd", "go", "testdata", "script")
	scripts, err := filepath.Glob(filepath.Join(scriptDir, "logos_*.txt"))
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range lint.Rules() {
		_, name, _ := strings.Cut(r.ID, "::")
		if name == "" {
			t.Errorf("rule %q has no name part", r.ID)
			continue
		}
		paths := lintFixturePaths(lintDir, names, name)
		if len(paths) == 0 {
			t.Errorf("rule %s has no fixture in test/logos/lint", r.ID)
			continue
		}
		if lintFixtureEvaluates(t, r, paths) {
			continue
		}
		if r.Check != nil {
			t.Errorf("rule %s: checked %d fixtures, the rule never fired", r.ID, len(paths))
			continue
		}
		if s := lintScriptAsserting(t, scripts, r.ID); s != "" {
			continue
		}
		t.Errorf("rule %s: checked %d fixtures with no finding and no logos script covers it", r.ID, len(paths))
	}
}

func lintFixturePaths(lintDir string, names map[string]bool, name string) []string {
	var paths []string
	for _, cand := range []string{name + ".lgo", name + ".go", name + "_go.go"} {
		if !names[cand] {
			continue
		}
		paths = append(paths, filepath.Join(lintDir, cand))
	}
	return paths
}

func lintTaggedRule(err error) (string, bool) {
	e, ok := err.(types2.Error)
	if !ok {
		return "", false
	}
	tag, _, ok := strings.Cut(e.Msg, "] ")
	if !ok {
		return "", false
	}
	rule, ok := strings.CutPrefix(tag, "[")
	if !ok {
		return "", false
	}
	if _, ok := lint.Lookup(rule); !ok {
		return "", false
	}
	return rule, true
}

func lintFixtureEvaluates(t *testing.T, r lint.Rule, paths []string) bool {
	t.Helper()
	for _, p := range paths {
		body, err := os.ReadFile(p)
		if err != nil {
			t.Errorf("fixture %s unreadable: %v", filepath.Base(p), err)
			continue
		}
		kind := lint.KindGo
		if strings.HasSuffix(p, ".lgo") {
			kind = lint.KindLgo
		}
		if !r.Go && kind == lint.KindGo {
			continue
		}
		results, tagged := lintCheckFixture(t, kind, filepath.Base(p), string(body))
		for _, got := range results {
			if got.Rule == r.ID {
				return true
			}
		}
		for _, tag := range tagged {
			if tag == r.ID {
				return true
			}
		}
	}
	return false
}

func lintCheckFixture(t *testing.T, kind lint.Kind, name, src string) ([]lint.Result, []string) {
	t.Helper()
	mode := syntax.CheckBranches
	if kind == lint.KindLgo {
		mode |= syntax.LogosMode
	}
	var perrs []error
	file, _ := syntax.Parse(syntax.NewFileBase(name), strings.NewReader(src), func(err error) {
		perrs = append(perrs, err)
	}, nil, mode)
	if len(perrs) > 0 || file == nil {
		t.Errorf("fixture %s does not parse: %v", name, perrs)
		return nil, nil
	}
	info := &types2.Info{
		StoreTypesInSyntax: true,
		Defs:               make(map[*syntax.Name]types2.Object),
		Uses:               make(map[*syntax.Name]types2.Object),
		Selections:         make(map[*syntax.SelectorExpr]*types2.Selection),
		Implicits:          make(map[syntax.Node]types2.Object),
		Scopes:             make(map[syntax.Node]*types2.Scope),
		Instances:          make(map[*syntax.Name]types2.Instance),
	}
	var errs []error
	var reported []lint.Reported
	var tagged []string
	conf := &types2.Config{
		Importer: newTestImporter(t),
		Error: func(err error) {
			if r, ok := lint.ReportedOf(err); ok {
				reported = append(reported, r)
				return
			}
			if rule, ok := lintTaggedRule(err); ok {
				tagged = append(tagged, rule)
				return
			}
			errs = append(errs, err)
		},
		Logos:       kind == lint.KindLgo,
		LogosUnions: kind == lint.KindLgo,
	}
	pkg, _ := conf.Check("p", []*syntax.File{file}, info)
	if len(errs) > 0 {
		t.Errorf("fixture %s does not check: %v", name, errs)
		return nil, tagged
	}
	policy, err := lint.ParsePolicy([]byte(`{"version":1,"groups":{"correctness":"forbid","concurrency":"forbid","slop":"forbid","complexity":"forbid","perf":"forbid","style":"forbid","suspicious":"forbid","portability":"forbid","pedantic":"forbid"}}`))
	if err != nil {
		t.Fatalf("enabling policy rejected: %v", err)
	}
	return lint.Run(&lint.Input{
		Pkg:      pkg,
		Info:     info,
		Files:    []*syntax.File{file},
		Sources:  map[*syntax.File][]byte{file: []byte(src)},
		Kind:     kind,
		Policy:   policy,
		Reported: reported,
	}), tagged
}

func lintScriptAsserting(t *testing.T, scripts []string, id string) string {
	t.Helper()
	for _, s := range scripts {
		body, err := os.ReadFile(s)
		if err != nil {
			t.Errorf("script %s unreadable: %v", filepath.Base(s), err)
			continue
		}
		for line := range strings.SplitSeq(string(body), "\n") {
			cmd := strings.TrimSpace(line)
			if cmd == "" || strings.HasPrefix(cmd, "!") || strings.HasPrefix(cmd, "#") {
				continue
			}
			if !strings.HasPrefix(cmd, "stderr") && !strings.HasPrefix(cmd, "stdout") && !strings.HasPrefix(cmd, "grep ") && !strings.HasPrefix(cmd, "cmp ") {
				continue
			}
			if strings.Contains(cmd, id) {
				return filepath.Base(s)
			}
		}
	}
	return ""
}
