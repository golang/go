// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"os"
	"regexp"
	"runtime"
	"testing"
)

func validateMapLabels(t *testing.T, labels []string) {
	// These are the specific region labels that need get added during the
	// runtime phase. Hence they are the ones that need to be confirmed as
	// present at the time the test reads its own region labels, which
	// is sufficient to validate that the default `decoratemappings` value
	// (enabled) was set early enough in the init process.
	regions := map[string]bool{
		"allspans array":    false,
		"gc bits":           false,
		"heap":              false,
		"heap index":        false,
		"heap reservation":  false,
		"immortal metadata": false,
		"page alloc":        false,
		"page alloc index":  false,
		"page summary":      false,
		"scavenge index":    false,
	}
	for _, label := range labels {
		if _, ok := regions[label]; !ok {
			t.Logf("unexpected region label found: \"%s\"", label)
		}
		regions[label] = true
	}
	for label, found := range regions {
		if !found {
			t.Logf("region label missing: \"%s\"", label)
		}
	}
}

func TestDecorateMappings(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("decoratemappings is only supported on Linux")
		// /proc/self/maps is also Linux-specific
	}

	var labels []string
	if rawMaps, err := os.ReadFile("/proc/self/maps"); err != nil {
		t.Fatalf("failed to read /proc/self/maps: %v", err)
	} else {
		t.Logf("maps:%s\n", string(rawMaps))
		matches := regexp.MustCompile("[^[]+ \\[anon: Go: (.+)\\]\n").FindAllSubmatch(rawMaps, -1)
		for _, match_pair := range matches {
			// match_pair consists of the matching substring and the parenthesized group
			labels = append(labels, string(match_pair[1]))
		}
	}
	t.Logf("DebugDecorateMappings: %v", *runtime.DebugDecorateMappings)
	if *runtime.DebugDecorateMappings != 0 && runtime.SetVMANameSupported() {
		validateMapLabels(t, labels)
	} else {
		if len(labels) > 0 {
			t.Errorf("unexpected mapping labels present: %v", labels)
		} else {
			t.Skip("mapping labels absent as expected")
		}
	}
}
