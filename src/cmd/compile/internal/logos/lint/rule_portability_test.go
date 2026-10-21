// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"internal/testenv"
	"slices"
	"strings"
	"testing"
)

// TestEnforcedElsewhereRules checks the registry entries of the rules that a
// phase before Run enforces under the same ID: the checker reports
// suspicious::unapplied_struct_default and suspicious::side_effect_before_sort,
// and the go command reports portability::alias_extension. Each entry is a
// loGo/s-only warning with no Check, so a //logos:allow directive naming it
// is not an allow_forbidden_rule error.
func TestEnforcedElsewhereRules(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	for _, tc := range []struct {
		id    string
		group Group
	}{
		{"suspicious::unapplied_struct_default", Suspicious},
		{"suspicious::side_effect_before_sort", Suspicious},
		{"portability::alias_extension", Portability},
	} {
		r, ok := Lookup(tc.id)
		if !ok {
			t.Errorf("%s is not registered", tc.id)
			continue
		}
		if r.Go || r.Check != nil || r.Group != tc.group {
			t.Errorf("%s entry = Go %v, Check set %v, Group %s; want a loGo/s-only entry with no Check in group %s", tc.id, r.Go, r.Check != nil, r.Group, tc.group)
		}
		if got := DefaultPolicy().Level(tc.id, KindLgo); got != Warn {
			t.Errorf("%s default level = %v, want warn", tc.id, got)
		}
		src := "package p\n\n//logos:allow(" + tc.id + ")\nfunc f() {}\n"
		if got := summarize(Run(check(t, KindLgo, "x.lgo", src))); len(got) != 0 {
			t.Errorf("%s: a directive naming the warn rule reported %q", tc.id, got)
		}
	}
}

// TestRegisteredSeverityAndScope checks the severity and scope of the two
// rules M9-S2 confirms rather than adds: pedantic::exhaustive_union_switch
// is allow by default and has a Check, and concurrency::unsupervised_goroutine
// is forbid and has none because the checker enforces it. Neither runs on Go
// packages.
func TestRegisteredSeverityAndScope(t *testing.T) {
	for _, tc := range []struct {
		id       string
		group    Group
		level    Severity
		hasCheck bool
	}{
		{"pedantic::exhaustive_union_switch", Pedantic, Allow, true},
		{"concurrency::unsupervised_goroutine", Concurrency, Forbid, false},
	} {
		r, ok := Lookup(tc.id)
		if !ok {
			t.Errorf("%s is not registered", tc.id)
			continue
		}
		if r.Go || r.Group != tc.group || (r.Check != nil) != tc.hasCheck {
			t.Errorf("%s entry = Go %v, Group %s, Check set %v; want loGo/s only, group %s, Check set %v", tc.id, r.Go, r.Group, r.Check != nil, tc.group, tc.hasCheck)
		}
		if got := DefaultPolicy().Level(tc.id, KindLgo); got != tc.level {
			t.Errorf("%s default level = %v, want %v", tc.id, got, tc.level)
		}
		if got := DefaultPolicy().Level(tc.id, KindGo); got != tc.level && (tc.level != Forbid || got != Deny) {
			t.Errorf("%s level in a Go package = %v", tc.id, got)
		}
	}
}

const serializedSrc = `package p

type A struct {
	N int    ` + "`json:\"n\"`" + ` // want
	M uint   ` + "`json:\"m\"`" + ` // want
	S string ` + "`json:\"s\"`" + `
	W int32  ` + "`json:\"w\"`" + `
	X int64  ` + "`json:\"x\"`" + `
	Y uint8  ` + "`json:\"y\"`" + `
	Z uint64 ` + "`json:\"z\"`" + `
}

type Untagged struct {
	N int
	M uint
}

type OtherKeys struct {
	N int ` + "`yaml:\"n\"`" + `
}

type Mixed struct {
	N int // want
	S string ` + "`json:\"s\"`" + `
}

type Alias = int
type Named int

type Kinds struct {
	A Alias ` + "`json:\"a\"`" + ` // want
	B Named ` + "`json:\"b\"`" + `
	C *int  ` + "`json:\"c\"`" + `
	D []int ` + "`json:\"d\"`" + `
	e int // want
}

type Skipped struct {
	N int ` + "`json:\"-\"`" + ` // want
}

type Outer struct {
	In struct {
		N int ` + "`json:\"n\"`" + ` // want
	} ` + "`json:\"in\"`" + `
}

type G[T any] struct {
	V T   ` + "`json:\"v\"`" + `
	N int ` + "`json:\"n\"`" + ` // want
}

//logos:allow(portability::serialized_unsized_int)
type Allowed struct {
	N int ` + "`json:\"n\"`" + `
}
`

const serializedDefaultsSrc = `package p

type Defaulted struct {
	Port int = 8080 ` + "`json:\"port\"`" + ` // want
	Host string = "h" ` + "`json:\"host\"`" + `
	Plain int = 1 // want
	Small int32 = 2
}

type DefaultedUntagged struct {
	Port int = 8080
}
`

// TestSerializedUnsizedIntDefaults covers fields with a default, whose type
// sits inside a field-default node of the loGo/s syntax tree.
func TestSerializedUnsizedIntDefaults(t *testing.T) {
	expect(t, KindLgo, nil, testSource{"x.lgo", serializedDefaultsSrc}, finding{"portability::serialized_unsized_int", Warn})
}

// TestSerializedUnsizedInt covers the detection in a loGo/s package: an int
// or uint field of a struct with a json-tagged field, never in a Go package.
func TestSerializedUnsizedInt(t *testing.T) {
	const id = "portability::serialized_unsized_int"
	expect(t, KindLgo, nil, testSource{"x.lgo", serializedSrc}, finding{id, Warn})
	if got := summarize(Run(check(t, KindGo, "x.go", serializedSrc))); len(got) != 0 {
		t.Errorf("Go package results = %q, want none", got)
	}
	r, ok := Lookup(id)
	if !ok || r.Go || r.Group != Portability || r.Check == nil {
		t.Fatalf("registry entry = %+v, %v", r, ok)
	}
}

// TestSerializedUnsizedIntSeverity covers the group severity: allow drops
// the finding and deny turns it into an error that the directive silences.
func TestSerializedUnsizedIntSeverity(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const id = "portability::serialized_unsized_int"
	for _, tc := range []struct {
		level string
		want  Severity
		none  bool
	}{
		{"warn", Warn, false},
		{"deny", Deny, false},
		{"allow", Allow, true},
	} {
		p, err := ParsePolicy([]byte(`{"version":1,"groups":{"portability":"` + tc.level + `"}}`))
		if err != nil {
			t.Fatal(err)
		}
		in := check(t, KindLgo, "x.lgo", serializedSrc)
		in.Policy = p
		got := summarize(Run(in))
		if tc.none {
			if len(got) != 0 {
				t.Errorf("%s: results %q, want none", tc.level, got)
			}
			continue
		}
		want := wantMarked(serializedSrc, id, tc.want)
		if !slices.Equal(got, want) {
			t.Errorf("%s: results\n got %q\nwant %q", tc.level, got, want)
		}
	}
}

// TestSerializedUnsizedIntSharedType checks that every name of a field list
// that shares one type is flagged on its own.
func TestSerializedUnsizedIntSharedType(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const src = "package p\n\ntype A struct {\n\tX, Y int `json:\"xy\"`\n\tZ, W int32 `json:\"zw\"`\n\tLo, Hi uint = 3 `json:\"r\"`\n}\n"
	got := summarize(Run(check(t, KindLgo, "x.lgo", src)))
	want := []string{
		"4 portability::serialized_unsized_int warn", "4 portability::serialized_unsized_int warn",
		"6 portability::serialized_unsized_int warn", "6 portability::serialized_unsized_int warn",
	}
	if !slices.Equal(got, want) {
		t.Errorf("results = %q, want %q", got, want)
	}
}

// TestSerializedUnsizedIntSpan checks that a finding spans the field name and
// names the type and the sized kinds to use.
func TestSerializedUnsizedIntSpan(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const src = "package p\n\ntype A struct {\n\tCount uint `json:\"c\"`\n}\n"
	rs := Run(check(t, KindLgo, "x.lgo", src))
	if len(rs) != 1 {
		t.Fatalf("results = %v, want one", rs)
	}
	r := rs[0]
	if r.Pos.Line() != 4 || r.Pos.Col() != 2 || r.End.Line() != 4 || r.End.Col() != 7 {
		t.Errorf("span = %d:%d-%d:%d, want 4:2-4:7", r.Pos.Line(), r.Pos.Col(), r.End.Line(), r.End.Col())
	}
	for _, part := range []string{"field Count has type uint", "4 bytes on a 32-bit target", "8 bytes in the binary layout", "int32", "int64", "uint32", "uint64"} {
		if !strings.Contains(r.Message, part) {
			t.Errorf("message %q lacks %q", r.Message, part)
		}
	}
}
