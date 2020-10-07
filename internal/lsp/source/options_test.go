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
			value: "dynamic",
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
			name:  "staticcheck",
			value: true,
			check: func(o Options) bool { return o.Staticcheck == true },
		},
		{
			name:  "codelens",
			value: map[string]interface{}{"generate": true},
			check: func(o Options) bool { return o.Codelens["generate"] },
		},
		{
			name:  "allExperiments",
			value: true,
			check: func(o Options) bool {
				return true // just confirm that we handle this setting
			},
		},
	}

	for _, test := range tests {
		var opts Options
		result := opts.set(test.name, test.value)
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
