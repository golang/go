// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package metrics_test

import (
	"bufio"
	"os"
	"regexp"
	"runtime/metrics"
	"strings"
	"testing"
)

func TestDescriptionNameFormat(t *testing.T) {
	r := regexp.MustCompile("^(?P<name>/[^:]+):(?P<unit>[^:*/]+(?:[*/][^:*/]+)*)$")
	descriptions := metrics.All()
	for _, desc := range descriptions {
		if !r.MatchString(desc.Name) {
			t.Errorf("metrics %q does not match regexp %s", desc.Name, r)
		}
	}
}

func extractMetricDocs(t *testing.T) map[string]string {
	f, err := os.Open("doc.go")
	if err != nil {
		t.Fatalf("failed to open doc.go in runtime/metrics package: %v", err)
	}
	const (
		stateSearch          = iota // look for list of metrics
		stateNextMetric             // look for next metric
		stateNextDescription        // build description
	)
	state := stateSearch
	s := bufio.NewScanner(f)
	result := make(map[string]string)
	var metric string
	var prevMetric string
	var desc strings.Builder
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		switch state {
		case stateSearch:
			if line == "Below is the full list of supported metrics, ordered lexicographically." {
				state = stateNextMetric
			}
		case stateNextMetric:
			// Ignore empty lines until we find a non-empty
			// one. This will be our metric name.
			if len(line) != 0 {
				prevMetric = metric
				metric = line
				if prevMetric > metric {
					t.Errorf("metrics %s and %s are out of lexicographical order", prevMetric, metric)
				}
				state = stateNextDescription
			}
		case stateNextDescription:
			if len(line) == 0 || line == `*/` {
				// An empty line means we're done.
				// Write down the description and look
				// for a new metric.
				result[metric] = desc.String()
				desc.Reset()
				state = stateNextMetric
			} else {
				// As long as we're seeing data, assume that's
				// part of the description and append it.
				if desc.Len() != 0 {
					// Turn previous newlines into spaces.
					desc.WriteString(" ")
				}
				desc.WriteString(line)
			}
		}
		if line == `*/` {
			break
		}
	}
	if state == stateSearch {
		t.Fatalf("failed to find supported metrics docs in %s", f.Name())
	}
	return result
}

func TestDescriptionDocs(t *testing.T) {
	docs := extractMetricDocs(t)
	descriptions := metrics.All()
	for _, d := range descriptions {
		want := d.Description
		got, ok := docs[d.Name]
		if !ok {
			t.Errorf("no docs found for metric %s", d.Name)
			continue
		}
		if got != want {
			t.Errorf("mismatched description and docs for metric %s", d.Name)
			t.Errorf("want: %q, got %q", want, got)
			continue
		}
	}
	if len(docs) > len(descriptions) {
	docsLoop:
		for name, _ := range docs {
			for _, d := range descriptions {
				if name == d.Name {
					continue docsLoop
				}
			}
			t.Errorf("stale documentation for non-existent metric: %s", name)
		}
	}
}
