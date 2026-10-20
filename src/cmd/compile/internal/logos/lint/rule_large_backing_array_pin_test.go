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

const pinSrc = `package p

func a(buf []byte) []byte {
	return buf[:10]
}

func b(s string) string {
	//logos:allow(perf::large_backing_array_pin)
	return s[:10]
}

func c(s string) string {
	return s[:10]
}
`

// pinKinds lists the file name of each kind of package.
var pinKinds = map[string]Kind{"x.go": KindGo, "x.lgo": KindLgo}

// TestLargeBackingArrayPinRegistered checks the scope of the rule: it has no
// Check, because escape analysis decides it, and it applies to Go packages.
func TestLargeBackingArrayPinRegistered(t *testing.T) {
	r, ok := Lookup("perf::large_backing_array_pin")
	if !ok || !r.Go || r.Check != nil || r.Group != Perf {
		t.Fatalf("registry entry = %+v, %v", r, ok)
	}
}

// TestRunPins covers the resolution of escape findings in both kinds of
// package: the default severity, the message of a subslice and of a
// substring, the position of each finding, and one result for a position
// that escape analysis reports twice.
func TestRunPins(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	for name, kind := range pinKinds {
		t.Run(name, func(t *testing.T) {
			in := check(t, kind, name, pinSrc)
			got := RunPins(in, []PinFinding{
				{File: name, Line: 4, Col: 10, Bytes: 65537},
				{File: name, Line: 4, Col: 10, Bytes: 65537},
				{File: name, Line: 13, Col: 10, Bytes: 70000, Substring: true},
			})
			if want := []string{"4 perf::large_backing_array_pin warn", "13 perf::large_backing_array_pin warn"}; !slices.Equal(summarize(got), want) {
				t.Fatalf("results = %q, want %q", summarize(got), want)
			}
			if m := got[0].Message; !strings.HasPrefix(m, "subslice of a 65537-byte backing array escapes") {
				t.Errorf("subslice message = %q", m)
			}
			if m := got[1].Message; !strings.HasPrefix(m, "substring of a 70000-byte backing string escapes") {
				t.Errorf("substring message = %q", m)
			}
		})
	}
}

// TestRunPinsDirective checks that //logos:allow(perf::large_backing_array_pin)
// above a statement silences a finding inside it, at warn and at deny, and
// that a finding in a statement without the directive survives.
func TestRunPinsDirective(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	for _, level := range []string{"warn", "deny"} {
		t.Run(level, func(t *testing.T) {
			p, err := ParsePolicy([]byte(`{"version":1,"groups":{"perf":"` + level + `"}}`))
			if err != nil {
				t.Fatal(err)
			}
			in := check(t, KindGo, "x.go", pinSrc)
			in.Policy = p
			got := RunPins(in, []PinFinding{
				{File: "x.go", Line: 4, Col: 10, Bytes: 65537},
				{File: "x.go", Line: 9, Col: 10, Bytes: 65537, Substring: true},
			})
			want := []string{"4 perf::large_backing_array_pin " + level}
			if !slices.Equal(summarize(got), want) {
				t.Errorf("results = %q, want %q", summarize(got), want)
			}
		})
	}
}

// TestRunPinsAllowed checks that perf = "allow" silences the rule and turns
// the escape phase off.
func TestRunPinsAllowed(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	p, err := ParsePolicy([]byte(`{"version":1,"groups":{"perf":"allow"}}`))
	if err != nil {
		t.Fatal(err)
	}
	in := check(t, KindGo, "x.go", pinSrc)
	if !PinActive(in) {
		t.Error("PinActive = false at the default severity")
	}
	in.Policy = p
	if PinActive(in) {
		t.Error("PinActive = true with perf = allow")
	}
	if got := RunPins(in, []PinFinding{{File: "x.go", Line: 4, Col: 10, Bytes: 65537}}); len(got) != 0 {
		t.Errorf("perf = allow reported %q", summarize(got))
	}
}

// TestRunPinsForeignFile checks that a finding in a file the package does
// not hold is dropped: it comes from a function of another package that the
// compiler inlined here, and that package's own lint reports it.
func TestRunPinsForeignFile(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	in := check(t, KindGo, "x.go", pinSrc)
	got := RunPins(in, []PinFinding{
		{File: "other.go", Line: 7, Col: 2, Bytes: 65537},
		{File: "x.go", Line: 4, Col: 10, Bytes: 65537},
	})
	if want := []string{"4 perf::large_backing_array_pin warn"}; !slices.Equal(summarize(got), want) {
		t.Errorf("results = %q, want %q", summarize(got), want)
	}
}
