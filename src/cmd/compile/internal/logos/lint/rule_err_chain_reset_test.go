// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"fmt"
	"internal/testenv"
	"slices"
	"strings"
	"testing"
)

// TestFormatVerbs covers the walk of a format string: the verb of each
// directive and the operand it formats, as package fmt assigns them.
func TestFormatVerbs(t *testing.T) {
	for _, tc := range []struct {
		format string
		want   []string
	}{
		{"", nil},
		{"plain", nil},
		{"%v", []string{"%v:0"}},
		{"a %s b %d", []string{"%s:0", "%d:1"}},
		{"100%% %v", []string{"%v:0"}},
		{"%+v %-5s % x %#v %05d", []string{"%+v:0", "%-5s:1", "% x:2", "%#v:3", "%05d:4"}},
		{"%10.3f %v", []string{"%10.3f:0", "%v:1"}},
		{"%*d %v", []string{"%*d:1", "%v:2"}},
		{"%.*f %v", []string{"%.*f:1", "%v:2"}},
		{"%*.*f %v", []string{"%*.*f:2", "%v:3"}},
		{"%[2]v %[1]d", []string{"%[2]v:1", "%[1]d:0"}},
		{"%[2]v %v", []string{"%[2]v:1", "%v:2"}},
		{"%v %[1]v %v", []string{"%v:0", "%[1]v:0", "%v:1"}},
		{"%6[2]d", []string{"%6[2]d:1"}},
		{"%[3]*.[2]*[1]f", []string{"%[3]*.[2]*[1]f:0"}},
		{"%[2]*d", []string{"%[2]*d:2"}},
		{"%[x]v %v", []string{"%[x]v:0", "%v:1"}},
		{"%[0]v", []string{"%[0]v:0"}},
		{"%é", []string{"%é:0"}},
		{"trailing %", nil},
		{"%-", nil},
		{"%w %w", []string{"%w:0", "%w:1"}},
	} {
		var got []string
		for _, v := range chainFormatVerbs(tc.format) {
			got = append(got, fmt.Sprintf("%s:%d", v.text, v.arg))
		}
		if !slices.Equal(got, tc.want) {
			t.Errorf("chainFormatVerbs(%q) = %q, want %q", tc.format, got, tc.want)
		}
	}
}

// TestFormatVerbOffsets checks that off and end bound the verb alone.
func TestFormatVerbOffsets(t *testing.T) {
	const format = "é %+5.2fé %[2]vé"
	for _, v := range chainFormatVerbs(format) {
		if got := format[v.off:v.end]; got != string(v.verb) {
			t.Errorf("verb %q spans %q", v.text, got)
		}
	}
}

const chainFixSrc = `package p

import (
	"errors"
	"fmt"
)

type codeErr struct{}

func (*codeErr) Error() string { return "code" }

var errBase = errors.New("base")

func verb(err error) error {
	return fmt.Errorf("read: %v", err)
}

func verbs(a, b error) error {
	return fmt.Errorf("%s then %+v", a, b)
}

func keepsOther(name string, err error) error {
	return fmt.Errorf("%s: %q (%d)", name, err, 3)
}

func call(err error) error {
	return fmt.Errorf("read: %s", err.Error())
}

func callAndVerb(a, b error) error {
	return fmt.Errorf("%v %s", a, b.Error())
}

func quoted(err error) error {
	return fmt.Errorf("read: %q", err.Error())
}

func concat(err error) error {
	return fmt.Errorf("read: " + err.Error())
}

func raw(err error) error {
	return fmt.Errorf(` + "`%v\\n`" + `, err)
}

func escaped(err error) error {
	return fmt.Errorf("tab\t%v \"q\" \u00e9", err)
}

func named(err error) error {
	const f = "named: %v"
	return fmt.Errorf(f, err)
}

func concrete(c *codeErr) error {
	return fmt.Errorf("c: %v %s", c, c.Error())
}

func multiline(err error) error {
	return fmt.Errorf(
		"read: %v",
		err,
	)
}

func nested(err error) error {
	return fmt.Errorf("outer: %v", fmt.Errorf("inner: %v", err))
}
`

// TestErrChainResetFix checks the finding and the fix text of each shape:
// the call text with the faulty verbs replaced by %w, or no fix when a fault
// has no verb to replace.
func TestErrChainResetFix(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	want := map[string][]string{
		"verb":        {`fmt.Errorf("read: %w", err)`},
		"verbs":       {`fmt.Errorf("%w then %+w", a, b)`},
		"keepsOther":  {`fmt.Errorf("%s: %w (%d)", name, err, 3)`},
		"call":        {`fmt.Errorf("read: %w", err)`},
		"callAndVerb": {`fmt.Errorf("%w %w", a, b)`},
		"quoted":      {""},
		"concat":      {""},
		"raw":         {"fmt.Errorf(`%w\\n`, err)"},
		"escaped":     {`fmt.Errorf("tab\t%w \"q\" é", err)`},
		"named":       {""},
		"concrete":    {`fmt.Errorf("c: %w %w", c, c)`},
		"multiline":   {`fmt.Errorf("read: %w", err)`},
		"nested":      {`fmt.Errorf("outer: %w", fmt.Errorf("inner: %v", err))`, `fmt.Errorf("inner: %w", err)`},
	}
	for _, kind := range []Kind{KindGo, KindLgo} {
		name := "x.go"
		if kind == KindLgo {
			name = "x.lgo"
		}
		got := chainFindingsByFunc(Run(check(t, kind, name, chainFixSrc)), chainFixSrc)
		for fn, fixes := range want {
			rs := got[fn]
			if len(rs) != len(fixes) {
				t.Errorf("%s: %s has %d findings, want %d", name, fn, len(rs), len(fixes))
				continue
			}
			for i, r := range rs {
				switch fix := fixes[i]; {
				case fix == "" && len(r.Fixes) != 0:
					t.Errorf("%s: %s: fix %q, want none", name, fn, r.Fixes[0].Replacement)
				case fix != "" && (len(r.Fixes) != 1 || r.Fixes[0].Replacement != fix || r.Fixes[0].InsertBeforeLine != 0):
					t.Errorf("%s: %s: fixes %+v, want one replacing the span with %q", name, fn, r.Fixes, fix)
				}
			}
		}
		for fn := range got {
			if _, ok := want[fn]; !ok {
				t.Errorf("%s: unexpected findings in %s", name, fn)
			}
		}
	}
}

// chainFindingsByFunc groups the err_chain_reset findings by the function of src
// that holds them, in position order.
func chainFindingsByFunc(rs []Result, src string) map[string][]Result {
	srcLines := strings.Split(src, "\n")
	out := make(map[string][]Result)
	for _, r := range rs {
		if r.Rule != "slop::err_chain_reset" {
			continue
		}
		for l := int(r.Pos.Line()) - 1; l >= 0; l-- {
			if rest, ok := strings.CutPrefix(srcLines[l], "func "); ok {
				fn, _, _ := strings.Cut(rest, "(")
				out[fn] = append(out[fn], r)
				break
			}
		}
	}
	return out
}

// TestErrChainResetMessage checks the message, span, severity, and
// provenance of one finding of each form.
func TestErrChainResetMessage(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	got := chainFindingsByFunc(Run(check(t, KindGo, "x.go", chainFixSrc)), chainFixSrc)
	verb, call := got["verb"][0], got["call"][0]
	if want := "fmt.Errorf formats error operand err with %v: use %w to keep the error chain"; verb.Message != want {
		t.Errorf("verb message = %q, want %q", verb.Message, want)
	}
	if want := "fmt.Errorf passes err.Error(): pass the error itself with %w to keep the error chain"; call.Message != want {
		t.Errorf("call message = %q, want %q", call.Message, want)
	}
	if got := spanText(chainFixSrc, verb); got != `fmt.Errorf("read: %v", err)` {
		t.Errorf("verb span = %q", got)
	}
	if got := verb.Provenance; got.InferredType != "error" || got.OriginExpr != `fmt.Errorf("read: %v", err)` || got.OriginLine != 15 {
		t.Errorf("verb provenance = %+v", got)
	}
	if got := got["concrete"][0].Provenance.InferredType; got != "*codeErr" {
		t.Errorf("concrete operand type = %q, want *codeErr", got)
	}
	if verb.Severity != Deny {
		t.Errorf("severity = %v, want deny", verb.Severity)
	}
}

// TestErrChainResetNone covers calls the rule leaves alone, with the shapes
// that look close to a finding.
func TestErrChainResetNone(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const src = `package p

import (
	"fmt"
	"log"
)

type notError struct{}

func (notError) Error(n int) string { return "" }

type wrongResult struct{}

func (wrongResult) Error() int { return 0 }

func errorf(format string, args ...any) error { return nil }

func w(a, b error) error { return fmt.Errorf("%w %w", a, b) }
func wIndexed(a error, n int) error { return fmt.Errorf("%[2]w %[1]d", n, a) }
func plain(n int, s string) error { return fmt.Errorf("%v %s", n, s) }
func anyValue(v any) error { return fmt.Errorf("%v", v) }
func spread(format string, args ...any) error { return fmt.Errorf(format, args...) }
func spreadConst(args ...any) error { return fmt.Errorf("%v", args...) }
func dynamic(format string, err error) error { return fmt.Errorf(format, err) }
func nilValue() error { return fmt.Errorf("%v", nil) }
func missing(err error) error { return fmt.Errorf("%v") }
func extra(err error) error { return fmt.Errorf("x", err) }
func noArgs() error { return fmt.Errorf("x") }
func sprintf(err error) string { return fmt.Sprintf("%v", err) }
func logf(err error) { log.Printf("%v", err.Error()) }
func other(err error) error { return errorf("%v", err) }
func method(n notError) error { return fmt.Errorf("%s", n.Error(1)) }
func result(n wrongResult) error { return fmt.Errorf("%d", n.Error()) }
func asString(err error) string { return err.Error() }
func derived(err error) error { s := err.Error(); return fmt.Errorf("%s", s) }
func errorNew(err error) error { return fmt.Errorf("%s", fmt.Sprint(err)) }
func indexOutOfRange(err error) error { return fmt.Errorf("%[3]v", err) }
`
	for _, kind := range []Kind{KindGo, KindLgo} {
		name := "x.go"
		if kind == KindLgo {
			name = "x.lgo"
		}
		for _, r := range Run(check(t, kind, name, src)) {
			if r.Rule == "slop::err_chain_reset" {
				t.Errorf("%s: unexpected finding on line %d: %s", name, r.Pos.Line(), r.Message)
			}
		}
	}
}

// TestErrChainResetAllow checks that //logos:allow silences a finding of the
// rule, and that the group severity governs it.
func TestErrChainResetAllow(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const src = `package p

import "fmt"

func a(err error) error {
	//logos:allow(slop::err_chain_reset)
	return fmt.Errorf("hidden: %v", err)
}

func b(err error) error {
	return fmt.Errorf("shown: %v", err)
}
`
	p, err := ParsePolicy([]byte(`{"version":1,"groups":{"slop":"warn"}}`))
	if err != nil {
		t.Fatal(err)
	}
	in := check(t, KindGo, "x.go", src)
	in.Policy = p
	got := summarize(Run(in))
	if want := []string{"11 slop::err_chain_reset warn"}; !slices.Equal(got, want) {
		t.Errorf("results = %q, want %q", got, want)
	}
	if got := summarize(Run(check(t, KindLgo, "x.lgo", src))); !slices.Equal(got, []string{"11 slop::err_chain_reset deny"}) {
		t.Errorf("lgo results = %q", got)
	}
	if !strings.Contains(src, "hidden") {
		t.Fatal("source lost its allowed call")
	}
}
