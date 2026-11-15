// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"errors"
	"slices"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

const covSliceDecls = `
type Creature interface{ Name() string }

type Named interface {
	Creature
	Weight() int
}

type Monster struct{ n string }

func (m *Monster) Name() string { return m.n }

type Rock struct{ n string }

func (r Rock) Name() string { return r.n }

func Names(cs const []Creature) int { return len(cs) }

func pair(a, b const []Creature) int { return len(a) + len(b) }
`

const covRTSource = `package rt

func CovSlice[I any, E any](s []E) []I {
	if s == nil {
		return nil
	}
	r := make([]I, len(s))
	for i, e := range s {
		var a any = e
		if a == nil {
			continue
		}
		r[i] = a.(I)
	}
	return r
}
`

type covRTImporter struct{ rt *types2.Package }

func (m covRTImporter) Import(path string) (*types2.Package, error) {
	if path != "logos/rt" {
		return nil, errors.New("unknown import " + path)
	}
	return m.rt, nil
}

func newCovRTImporter(t *testing.T) types2.Importer {
	t.Helper()
	f, err := syntax.Parse(syntax.NewFileBase("rt.go"), strings.NewReader(covRTSource), func(err error) { t.Fatal(err) }, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	conf := types2.Config{Error: func(err error) { t.Fatal(err) }}
	pkg, err := conf.Check("logos/rt", []*syntax.File{f}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return covRTImporter{rt: pkg}
}

func covSliceCheck(t *testing.T, imp types2.Importer, imports, body string) (*syntax.File, []byte, *types2.Package, *types2.Info, []error) {
	t.Helper()
	data := []byte("package covtest\n" + imports + covSliceDecls + body)
	file, err := syntax.Parse(syntax.NewFileBase("cov.lgo"), bytes.NewReader(data), func(err error) { t.Fatalf("parse: %v", err) }, nil, syntax.LogosMode)
	if err != nil {
		t.Fatal(err)
	}
	info := &types2.Info{
		Types:      make(map[syntax.Expr]types2.TypeAndValue),
		Defs:       make(map[*syntax.Name]types2.Object),
		Uses:       make(map[*syntax.Name]types2.Object),
		Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
		Instances:  make(map[*syntax.Name]types2.Instance),
	}
	var errs []error
	conf := types2.Config{Importer: imp, Error: func(err error) { errs = append(errs, err) }, Logos: true}
	pkg, _ := conf.Check("covtest", []*syntax.File{file}, info)
	return file, data, pkg, info, errs
}

func newCovTestImporter(t *testing.T) types2.Importer {
	return newTestImporter(t)
}

func covLower(t *testing.T, file *syntax.File, data []byte, pkg *types2.Package, info *types2.Info) (*fileLower, string) {
	t.Helper()
	fl := &fileLower{
		pkg:   &packageLower{info: info, tpkg: pkg},
		file:  file,
		src:   data,
		base:  file.Pos().FileBase(),
		start: lineStarts(data),
	}
	fl.lowerConstSlices()
	out, err := applySplices(data, slices.Clone(fl.splices))
	if err != nil {
		t.Fatalf("applySplices: %v", err)
	}
	return fl, string(out)
}

func TestLowerConstSliceLeavesPlainArguments(t *testing.T) {
	file, data, pkg, info, errs := covSliceCheck(t, newCovTestImporter(t), "", `
func use(cs []Creature, cv const []Creature) int {
	return Names(cs) + Names(nil) + Names(cv) + Names([]Creature{})
}
`)
	if len(errs) > 0 {
		t.Fatalf("check: %v", errs)
	}
	if len(info.LogosConstCov) != 0 {
		t.Fatalf("recorded %d covariant arguments, want none", len(info.LogosConstCov))
	}
	fl, text := covLower(t, file, data, pkg, info)
	if fl.needRT || len(fl.splices) != 0 || text != string(data) {
		t.Errorf("lowering changed a file without covariant arguments:\n%s", text)
	}
}

func TestLowerConstSliceTypeParameterProof(t *testing.T) {
	file, data, pkg, info, errs := covSliceCheck(t, newCovTestImporter(t), "", `
func use[T Creature](ts []T) int { return Names(ts) }
`)
	if len(errs) > 0 {
		t.Fatalf("check: %v", errors.Join(errs...))
	}
	if len(info.LogosConstCov) != 1 {
		t.Fatalf("recorded %d covariant arguments, want 1", len(info.LogosConstCov))
	}
	_, text := covLower(t, file, data, pkg, info)
	if want := "Names(_logos_rt.CovSlice[Creature](ts))"; !strings.Contains(text, want) {
		t.Errorf("eject text lacks %q:\n%s", want, text)
	}
}

func TestConstSliceNotAdmitted(t *testing.T) {
	for name, tc := range map[string]struct{ body, want string }{
		"valueLacksPointerMethods": {
			`func use(vs []Monster) int { return Names(vs) }`,
			"const_view_escape",
		},
		"unrelatedElement": {
			`func use(xs []int) int { return Names(xs) }`,
			"const_view_escape",
		},
		"constSource": {
			`func use(cm const []*Monster) int { return Names(cm) }`,
			"const_view_escape",
		},
		"variable": {
			`func use(ms []*Monster) { var cv const []Creature = ms; _ = cv }`,
			"const_view_escape",
		},
		"assignment": {
			`func use(ms []*Monster, cv const []Creature) { cv = ms; _ = cv }`,
			"const_view_escape",
		},
		"result": {
			`func use(ms []*Monster) const []Creature { return ms }`,
			"const_view_escape",
		},
		"unprovedTypeParameterElement": {
			`func use[T any](ts []T) int { return Names(ts) }`,
			"const_view_escape",
		},
		"spread": {
			`func two() ([]*Monster, []*Monster) { return nil, nil }
func mixed(a const []Creature, b []*Monster) int { return len(a) + len(b) }
func use() int { return mixed(two()) }`,
			"const_view_escape",
		},
		"mutableParameter": {
			`func plain(cs []Creature) int { return len(cs) }
func use(ms []*Monster) int { return plain(ms) }`,
			"cannot use",
		},
		"mutableVariable": {
			`func use(ms []*Monster) []Creature { var xs []Creature = ms; return xs }`,
			"cannot use",
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, _, _, info, errs := covSliceCheck(t, newCovTestImporter(t), "", tc.body)
			if len(errs) == 0 {
				t.Fatal("check accepted the program")
			}
			if got := errors.Join(errs...).Error(); !strings.Contains(got, tc.want) {
				t.Errorf("errors %q lack %q", got, tc.want)
			}
			if len(info.LogosConstCov) != 0 {
				t.Errorf("recorded %d covariant arguments for a rejected program", len(info.LogosConstCov))
			}
		})
	}
}

func TestCovSliceChecksAreLogosOnly(t *testing.T) {
	imp := newCovRTImporter(t)
	const src = `package covtest

import rt "logos/rt"

type Creature interface{ Name() string }

type Monster struct{ n string }

func (m *Monster) Name() string { return m.n }

func direct(vs []Monster) []Creature { return rt.CovSlice[Creature](vs) }

func mutable(ms []*Monster) []Creature { return ms }
`
	file, err := syntax.Parse(syntax.NewFileBase("plain.go"), strings.NewReader(src), func(err error) { t.Fatalf("parse: %v", err) }, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	info := &types2.Info{
		Types: make(map[syntax.Expr]types2.TypeAndValue),
		Defs:  make(map[*syntax.Name]types2.Object),
		Uses:  make(map[*syntax.Name]types2.Object),
	}
	var errs []error
	conf := types2.Config{Importer: imp, Error: func(err error) { errs = append(errs, err) }}
	conf.Check("covtest", []*syntax.File{file}, info)
	if len(errs) != 1 || !strings.Contains(errs[0].Error(), "cannot use ms") {
		t.Fatalf("plain Go check errors = %v, want only the mutable assignment", errs)
	}
	terr, ok := errs[0].(types2.Error)
	if !ok {
		t.Fatalf("plain Go error has type %T, want types2.Error", errs[0])
	}
	if got := terr.Pos.RelFilename(); got != "plain.go" {
		t.Errorf("plain Go error file = %q, want plain.go", got)
	}
	wantLine := uint(0)
	for i, line := range strings.Split(src, "\n") {
		if strings.Contains(line, "return ms") {
			wantLine = uint(i + 1)
		}
	}
	if terr.Pos.RelLine() != wantLine {
		t.Errorf("plain Go error line = %d, want %d", terr.Pos.RelLine(), wantLine)
	}
	if strings.Contains(errs[0].Error(), "CovSlice") {
		t.Errorf("plain Go error %q names CovSlice, want the plain assignment", errs[0])
	}
	if len(info.LogosConstCov) != 0 {
		t.Errorf("plain Go recorded %d covariant arguments", len(info.LogosConstCov))
	}
}

func TestCovSliceDirectCall(t *testing.T) {
	imp := newCovRTImporter(t)
	const imports = "\nimport rt \"logos/rt\"\n"
	for name, tc := range map[string]struct{ body, want string }{
		"concreteElement": {
			`func use(ms []*Monster) []Creature { return rt.CovSlice[Creature](ms) }`, "",
		},
		"explicitElement": {
			`func use(ms []*Monster) []Creature { return rt.CovSlice[Creature, *Monster](ms) }`, "",
		},
		"interfaceElement": {
			`func use(ns []Named) []Creature { return rt.CovSlice[Creature](ns) }`, "",
		},
		"provedTypeParameterElement": {
			`func use[T Creature](ts []T) []Creature { return rt.CovSlice[Creature](ts) }`, "",
		},
		"unprovedTypeParameterElement": {
			`func use[T any](ts []T) []Creature { return rt.CovSlice[Creature](ts) }`,
			"element type T does not implement interface Creature",
		},
		"nonInterfaceTarget": {
			`func use(xs []int) []int { return rt.CovSlice[int](xs) }`,
			"int is not an interface",
		},
		"typeParameterTarget": {
			`func use[I any, T any](ts []T) []I { return rt.CovSlice[I](ts) }`,
			"I is not an interface",
		},
		"valueLacksPointerMethods": {
			`func use(vs []Monster) []Creature { return rt.CovSlice[Creature](vs) }`,
			"element type Monster does not implement interface Creature",
		},
		"explicitValueLacksPointerMethods": {
			`func use(vs []Monster) []Creature { return rt.CovSlice[Creature, Monster](vs) }`,
			"element type Monster does not implement interface Creature",
		},
		"interfaceLacksMethods": {
			`func use(cs []Creature) []Named { return rt.CovSlice[Named](cs) }`,
			"element type Creature does not implement interface Named",
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, _, _, info, errs := covSliceCheck(t, imp, imports, tc.body)
			prefix := "package covtest\n" + imports + covSliceDecls
			wantLine := uint(strings.Count(prefix, "\n") + 1)
			got := errors.Join(errs...)
			switch {
			case tc.want == "" && got != nil:
				t.Fatalf("check rejected the call: %v", got)
			case tc.want != "" && (got == nil || !strings.Contains(got.Error(), tc.want)):
				t.Fatalf("errors %v lack %q", got, tc.want)
			}
			if tc.want == "" {
				if len(info.LogosConstCov) != 0 {
					t.Fatalf("accepted call recorded %d covariant arguments, want none for a direct call", len(info.LogosConstCov))
				}
				return
			}
			if len(errs) != 1 {
				t.Fatalf("got %d errors %v, want one naming %q", len(errs), got, tc.want)
			}
			terr, ok := errs[0].(types2.Error)
			if !ok {
				t.Fatalf("error has type %T, want types2.Error", errs[0])
			}
			if got := terr.Pos.RelFilename(); got != "cov.lgo" {
				t.Errorf("error file = %q, want cov.lgo", got)
			}
			if terr.Pos.RelLine() != wantLine {
				t.Errorf("error line = %d, want %d", terr.Pos.RelLine(), wantLine)
			}
			if wantCol := uint(strings.LastIndex(tc.body, "(") + 2); terr.Pos.RelCol() != wantCol {
				t.Errorf("error column = %d, want %d", terr.Pos.RelCol(), wantCol)
			}
			if len(info.LogosConstCov) != 0 {
				t.Errorf("rejected call recorded %d covariant arguments", len(info.LogosConstCov))
			}
		})
	}
}
