// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"internal/testenv"
	"slices"
	"testing"
)

const timeAfterSrc = `package p

import "time"

type clock struct{}

func (clock) After(d time.Duration) <-chan time.Time { return nil }

func After(d time.Duration) <-chan time.Time { return nil }

func a(d time.Duration, n int, xs []int) {
	<-time.After(d)
	for i := 0; i < n; i++ {
		<-time.After(d) // want
	}
	for range xs {
		<-time.After(d) // want
	}
	for t := time.After(d); n > 0; n-- {
		<-t
	}
	for range time.After(d) {
		break
	}
	for {
		for {
			<-time.After(d) // want
			break
		}
		func() {
			<-time.After(d) // want
		}()
		break
	}
	f := func() {
		for {
			<-time.After(d) // want
		}
	}
	_ = f
	var c clock
	now := time.Now()
	for {
		_ = now.After(now)
		<-c.After(d)
		<-After(d)
		_ = time.Tick
		g := time.After
		_ = g
		break
	}
	for {
		//logos:allow(perf::time_after_in_loop)
		<-time.After(d)
		break
	}
}
`

// timeAfterResults runs the rules over src and keeps the findings of
// perf::time_after_in_loop. The fixture also receives from time.After outside
// a done select, which correctness::unbounded_channel_op reports (SPEC Sec
// 1.2; test/logos/lint/unbounded_channel_op.lgo flags the same receive). That
// rule has its own tests, so these tests filter by rule as the other rule
// tests do.
func timeAfterResults(t *testing.T, kind Kind, p *Policy, name, src string) []Result {
	t.Helper()
	in := check(t, kind, name, src)
	in.Policy = p
	var out []Result
	for _, r := range Run(in) {
		if r.Rule == "perf::time_after_in_loop" {
			out = append(out, r)
		}
	}
	return out
}

// expectTimeAfter compares the findings of perf::time_after_in_loop over src
// at severity sev with the "// want" lines of src.
func expectTimeAfter(t *testing.T, kind Kind, p *Policy, name, src string, sev Severity) {
	t.Helper()
	testenv.MustHaveGoBuild(t)
	got := summarize(timeAfterResults(t, kind, p, name, src))
	want := wantMarked(src, "perf::time_after_in_loop", sev)
	if !slices.Equal(got, want) {
		t.Errorf("%s results:\n got %q\nwant %q", name, got, want)
	}
}

// TestTimeAfterInLoop covers the detection, the loop header, closures, the
// lookalikes that are not time.After, and the directive, in both kinds of
// package.
func TestTimeAfterInLoop(t *testing.T) {
	for name, kind := range map[string]Kind{"x.go": KindGo, "x.lgo": KindLgo} {
		t.Run(name, func(t *testing.T) {
			expectTimeAfter(t, kind, nil, name, timeAfterSrc, Warn)
		})
	}
}

// TestTimeAfterInLoopSeverity covers the configured severities of the rule.
func TestTimeAfterInLoopSeverity(t *testing.T) {
	for sev, level := range map[string]Severity{"deny": Deny, "warn": Warn} {
		t.Run(sev, func(t *testing.T) {
			p, err := ParsePolicy([]byte(`{"version":1,"groups":{"perf":"` + sev + `"}}`))
			if err != nil {
				t.Fatal(err)
			}
			expectTimeAfter(t, KindGo, p, "x.go", timeAfterSrc, level)
		})
	}
	t.Run("allow", func(t *testing.T) {
		testenv.MustHaveGoBuild(t)
		p, err := ParsePolicy([]byte(`{"version":1,"groups":{"perf":"allow"}}`))
		if err != nil {
			t.Fatal(err)
		}
		if got := summarize(timeAfterResults(t, KindGo, p, "x.go", timeAfterSrc)); len(got) != 0 {
			t.Errorf("perf = allow reported %q", got)
		}
	})
}

// TestTimeAfterInLoopSpan checks that the span of a finding covers its whole
// call.
func TestTimeAfterInLoopSpan(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const src = `package p

import "time"

func a(d time.Duration) {
	for {
		<-time.After(time.Duration(2) * d)
		<-time.After(
			d,
		)
	}
}
`
	want := []string{"time.After(time.Duration(2) * d)", "time.After(\n\t\t\td,\n\t\t)"}
	var got []string
	for _, r := range timeAfterResults(t, KindGo, nil, "x.go", src) {
		got = append(got, spanText(src, r))
	}
	if !slices.Equal(got, want) {
		t.Errorf("spans = %q, want %q", got, want)
	}
}
