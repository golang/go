// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package driverutil

import (
	"fmt"
	"net/url"

	"golang.org/x/tools/go/analysis"
)

// ResolveURL resolves the URL field for a Diagnostic from an Analyzer
// and returns the URL. See Diagnostic.URL for details.
func ResolveURL(a *analysis.Analyzer, d analysis.Diagnostic) (string, error) {
	if d.URL == "" && d.Category == "" && a.URL == "" {
		return "", nil // do nothing
	}
	raw := d.URL
	if d.URL == "" && d.Category != "" {
		raw = "#" + d.Category
	}
	u, err := url.Parse(raw)
	if err != nil {
		return "", fmt.Errorf("invalid Diagnostic.URL %q: %s", raw, err)
	}
	base, err := url.Parse(a.URL)
	if err != nil {
		return "", fmt.Errorf("invalid Analyzer.URL %q: %s", a.URL, err)
	}
	return base.ResolveReference(u).String(), nil
}
