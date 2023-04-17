// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package analysisflags_test

import (
	"fmt"
	"strings"
	"testing"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/internal/analysisflags"
)

func TestResolveURLs(t *testing.T) {
	// TestResolveURL test the 12 different combinations for how URLs can be resolved
	// when Analyzer.URL are Diagnostic.Category are empty or non-empty, and when
	// Diagnostic.URL is empty, absolute or relative.

	aURL := &analysis.Analyzer{URL: "https://analyzer.example"}
	noURL := &analysis.Analyzer{URL: ""}
	tests := []struct {
		analyzer   *analysis.Analyzer
		diagnostic analysis.Diagnostic
		want       string
	}{
		{noURL, analysis.Diagnostic{Category: "", URL: ""}, ""},
		{noURL, analysis.Diagnostic{Category: "", URL: "#relative"}, "#relative"},
		{noURL, analysis.Diagnostic{Category: "", URL: "https://absolute.diagnostic"}, "https://absolute.diagnostic"},
		{noURL, analysis.Diagnostic{Category: "category", URL: ""}, "#category"},
		{noURL, analysis.Diagnostic{Category: "category", URL: "#relative"}, "#relative"},
		{noURL, analysis.Diagnostic{Category: "category", URL: "https://absolute.diagnostic"}, "https://absolute.diagnostic"},
		{aURL, analysis.Diagnostic{Category: "", URL: ""}, "https://analyzer.example"},
		{aURL, analysis.Diagnostic{Category: "", URL: "#relative"}, "https://analyzer.example#relative"},
		{aURL, analysis.Diagnostic{Category: "", URL: "https://absolute.diagnostic"}, "https://absolute.diagnostic"},
		{aURL, analysis.Diagnostic{Category: "category", URL: ""}, "https://analyzer.example#category"},
		{aURL, analysis.Diagnostic{Category: "category", URL: "#relative"}, "https://analyzer.example#relative"},
		{aURL, analysis.Diagnostic{Category: "category", URL: "https://absolute.diagnostic"}, "https://absolute.diagnostic"},
	}
	for _, c := range tests {
		got, err := analysisflags.ResolveURL(c.analyzer, c.diagnostic)
		if err != nil {
			t.Errorf("Unexpected error from ResolveURL %s", err)
		} else if got != c.want {
			t.Errorf("ResolveURL(%q,%v)=%q. want %s", c.analyzer.URL, c.diagnostic, got, c.want)
		}
	}
}

func TestResolveURLErrors(t *testing.T) {
	tests := []struct {
		analyzer   *analysis.Analyzer
		diagnostic analysis.Diagnostic
		want       string
	}{
		{&analysis.Analyzer{URL: ":not a url"}, analysis.Diagnostic{Category: "", URL: "#relative"}, "invalid Analyzer.URL"},
		{&analysis.Analyzer{URL: "https://analyzer.example"}, analysis.Diagnostic{Category: "", URL: ":not a URL"}, "invalid Diagnostic.URL"},
	}
	for _, c := range tests {
		_, err := analysisflags.ResolveURL(c.analyzer, c.diagnostic)
		if got := fmt.Sprint(err); !strings.HasPrefix(got, c.want) {
			t.Errorf("ResolveURL(%q, %q) expected an error starting with %q. got %q", c.analyzer.URL, c.diagnostic.URL, c.want, got)
		}
	}
}
