// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package config

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// TestParseLints covers the [lints] and [lints.thresholds] tables.
func TestParseLints(t *testing.T) {
	f, err := Parse("logos.toml", []byte(`[lints]
slop = "warn"
pedantic = "deny"
correctness = "forbid"

[lints.thresholds]
max_function_lines = 80
max_nesting_depth = 2
`))
	if err != nil {
		t.Fatal(err)
	}
	want := Lints{
		Groups:     map[string]string{"slop": "warn", "pedantic": "deny", "correctness": "forbid"},
		Thresholds: map[string]int{"max_function_lines": 80, "max_nesting_depth": 2},
	}
	if !reflect.DeepEqual(f.Lints, want) {
		t.Errorf("Lints = %+v, want %+v", f.Lints, want)
	}
}

// TestParseLintsErrors covers every rejected [lints] shape.
func TestParseLintsErrors(t *testing.T) {
	for _, tc := range []struct{ name, data, want string }{
		{"lowered correctness", "[lints]\ncorrectness = \"warn\"\n", `logos.toml:2:15: [configuration::forbid_lowered] correctness cannot be lowered below forbid`},
		{"lowered concurrency", "[lints]\nconcurrency = \"allow\"\n", `logos.toml:2:15: [configuration::forbid_lowered] concurrency cannot be lowered below forbid`},
		{"bad severity", "[lints]\nslop = \"loud\"\n", `logos.toml:2:8: invalid slop value "loud" (want forbid, deny, warn, or allow)`},
		{"group outside table", "[project]\nslop = \"warn\"\n", `logos.toml:2:1: unknown key "slop"`},
		{"threshold outside table", "[lints]\nmax_function_lines = 80\n", `logos.toml:2:1: unknown key "max_function_lines"`},
		{"unknown threshold", "[lints.thresholds]\nmax_other = 3\n", `logos.toml:2:1: unknown key "max_other"`},
		{"zero threshold", "[lints.thresholds]\nmax_function_lines = 0\n", `logos.toml:2:22: invalid max_function_lines value "0" (want a positive integer)`},
		{"string threshold", "[lints.thresholds]\nmax_function_lines = \"80\"\n", `logos.toml:2:22: want integer for key "max_function_lines"`},
		{"missing threshold", "[lints.thresholds]\nmax_function_lines =\n", `logos.toml:2:21: expected value for key "max_function_lines"`},
		{"integer group", "[lints]\nslop = 3\n", `logos.toml:2:8: want string for key "slop"`},
		{"duplicate table", "[lints]\n[lints]\n", `logos.toml:2:1: duplicate table "lints"`},
		{"unknown nested table", "[lints.other]\n", `logos.toml:1:1: unknown table "lints.other"`},
	} {
		_, err := Parse("logos.toml", []byte(tc.data))
		if err == nil || err.Error() != tc.want {
			t.Errorf("%s: error = %v, want %q", tc.name, err, tc.want)
		}
	}
}

// TestPolicyJSON checks the -logos.cfg schema and its determinism.
func TestPolicyJSON(t *testing.T) {
	l := Lints{
		Groups:     map[string]string{"slop": "warn", "complexity": "allow", "perf": "deny"},
		Thresholds: map[string]int{"max_function_lines": 80, "max_function_params": 6},
	}
	first, err := l.PolicyJSON(PedanticDefault)
	if err != nil {
		t.Fatal(err)
	}
	for range 20 {
		again, err := l.PolicyJSON(PedanticDefault)
		if err != nil || string(again) != string(first) {
			t.Fatalf("PolicyJSON not deterministic: %s vs %s (%v)", first, again, err)
		}
	}
	const want = `{"version":1,"groups":{"complexity":"allow","perf":"deny","slop":"warn"},"thresholds":{"max_function_lines":80,"max_function_params":6}}`
	if string(first) != want {
		t.Errorf("PolicyJSON = %s, want %s", first, want)
	}
}

// TestPolicyJSONPedantic checks the --pedantic override of the pedantic key.
func TestPolicyJSONPedantic(t *testing.T) {
	for _, tc := range []struct {
		name   string
		lints  Lints
		mode   PedanticMode
		expect string
	}{
		{"flag raises allow", Lints{}, PedanticOn, "warn"},
		{"flag keeps stricter", Lints{Groups: map[string]string{"pedantic": "deny"}}, PedanticOn, "deny"},
		{"flag false forces allow", Lints{Groups: map[string]string{"pedantic": "deny"}}, PedanticOff, "allow"},
		{"no flag keeps config", Lints{Groups: map[string]string{"pedantic": "warn"}}, PedanticDefault, "warn"},
	} {
		data, err := tc.lints.PolicyJSON(tc.mode)
		if err != nil {
			t.Fatal(err)
		}
		var got lintPolicy
		if err := json.Unmarshal(data, &got); err != nil {
			t.Fatal(err)
		}
		if got.Groups["pedantic"] != tc.expect {
			t.Errorf("%s: pedantic = %q, want %q (%s)", tc.name, got.Groups["pedantic"], tc.expect, data)
		}
	}
	if def, _ := (Lints{}).PolicyJSON(PedanticDefault); strings.Contains(string(def), "pedantic") {
		t.Errorf("no key and no flag must leave pedantic to the compiler default: %s", def)
	}
}
