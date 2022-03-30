// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"testing"
	"time"
)

func TestSetOption(t *testing.T) {
	tests := []struct {
		name      string
		value     interface{}
		wantError bool
		check     func(Options) bool
	}{
		{
			name:  "symbolStyle",
			value: "Dynamic",
			check: func(o Options) bool { return o.SymbolStyle == DynamicSymbols },
		},
		{
			name:      "symbolStyle",
			value:     "",
			wantError: true,
			check:     func(o Options) bool { return o.SymbolStyle == "" },
		},
		{
			name:      "symbolStyle",
			value:     false,
			wantError: true,
			check:     func(o Options) bool { return o.SymbolStyle == "" },
		},
		{
			name:  "symbolMatcher",
			value: "caseInsensitive",
			check: func(o Options) bool { return o.SymbolMatcher == SymbolCaseInsensitive },
		},
		{
			name:  "completionBudget",
			value: "2s",
			check: func(o Options) bool { return o.CompletionBudget == 2*time.Second },
		},
		{
			name:      "staticcheck",
			value:     true,
			check:     func(o Options) bool { return o.Staticcheck == true },
			wantError: true, // o.StaticcheckSupported is unset
		},
		{
			name:  "codelenses",
			value: map[string]interface{}{"generate": true},
			check: func(o Options) bool { return o.Codelenses["generate"] },
		},
		{
			name:  "allExperiments",
			value: true,
			check: func(o Options) bool {
				return true // just confirm that we handle this setting
			},
		},
		{
			name:  "hoverKind",
			value: "FullDocumentation",
			check: func(o Options) bool {
				return o.HoverKind == FullDocumentation
			},
		},
		{
			name:  "hoverKind",
			value: "NoDocumentation",
			check: func(o Options) bool {
				return o.HoverKind == NoDocumentation
			},
		},
		{
			name:  "hoverKind",
			value: "SingleLine",
			check: func(o Options) bool {
				return o.HoverKind == SingleLine
			},
		},
		{
			name:  "hoverKind",
			value: "Structured",
			check: func(o Options) bool {
				return o.HoverKind == Structured
			},
		},
		{
			name:  "ui.documentation.hoverKind",
			value: "Structured",
			check: func(o Options) bool {
				return o.HoverKind == Structured
			},
		},
		{
			name:  "matcher",
			value: "Fuzzy",
			check: func(o Options) bool {
				return o.Matcher == Fuzzy
			},
		},
		{
			name:  "matcher",
			value: "CaseSensitive",
			check: func(o Options) bool {
				return o.Matcher == CaseSensitive
			},
		},
		{
			name:  "matcher",
			value: "CaseInsensitive",
			check: func(o Options) bool {
				return o.Matcher == CaseInsensitive
			},
		},
		{
			name:  "env",
			value: map[string]interface{}{"testing": "true"},
			check: func(o Options) bool {
				v, found := o.Env["testing"]
				return found && v == "true"
			},
		},
		{
			name:      "env",
			value:     []string{"invalid", "input"},
			wantError: true,
			check: func(o Options) bool {
				return o.Env == nil
			},
		},
		{
			name:  "directoryFilters",
			value: []interface{}{"-node_modules", "+project_a"},
			check: func(o Options) bool {
				return len(o.DirectoryFilters) == 2
			},
		},
		{
			name:      "directoryFilters",
			value:     []interface{}{"invalid"},
			wantError: true,
			check: func(o Options) bool {
				return len(o.DirectoryFilters) == 0
			},
		},
		{
			name:      "directoryFilters",
			value:     []string{"-invalid", "+type"},
			wantError: true,
			check: func(o Options) bool {
				return len(o.DirectoryFilters) == 0
			},
		},
		{
			name: "annotations",
			value: map[string]interface{}{
				"Nil":      false,
				"noBounds": true,
			},
			wantError: true,
			check: func(o Options) bool {
				return !o.Annotations[Nil] && !o.Annotations[Bounds]
			},
		},
	}

	for _, test := range tests {
		var opts Options
		result := opts.set(test.name, test.value, map[string]struct{}{})
		if (result.Error != nil) != test.wantError {
			t.Fatalf("Options.set(%q, %v): result.Error = %v, want error: %t", test.name, test.value, result.Error, test.wantError)
		}
		// TODO: this could be made much better using cmp.Diff, if that becomes
		// available in this module.
		if !test.check(opts) {
			t.Errorf("Options.set(%q, %v): unexpected result %+v", test.name, test.value, opts)
		}
	}
}
