// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips140

import (
	"crypto/sha256"
	"flag"
	"fmt"
	"internal/testenv"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

var update = flag.Bool("update", false, "update GOROOT/lib/fips140/fips140.sum")

func TestSums(t *testing.T) {
	lib := filepath.Join(testenv.GOROOT(t), "lib/fips140")
	file := filepath.Join(lib, "fips140.sum")
	sums, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.SplitAfter(string(sums), "\n")

	zips, err := filepath.Glob(filepath.Join(lib, "*.zip"))
	if err != nil {
		t.Fatal(err)
	}

	format := func(name string, sum [32]byte) string {
		return fmt.Sprintf("%s %x\n", name, sum[:])
	}

	want := make(map[string]string)
	for _, zip := range zips {
		data, err := os.ReadFile(zip)
		if err != nil {
			t.Fatal(err)
		}
		name := filepath.Base(zip)
		want[name] = format(name, sha256.Sum256(data))
	}

	// Process diff, deleting or correcting stale lines.
	var diff []string
	have := make(map[string]bool)
	for i, line := range lines {
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "#") || line == "\n" {
			// comment, preserve
			diff = append(diff, " "+line)
			continue
		}
		name, _, _ := strings.Cut(line, " ")
		if want[name] == "" {
			lines[i] = ""
			diff = append(diff, "-"+line)
			continue
		}
		have[name] = true
		fixed := want[name]
		delete(want, name)
		if line == fixed {
			diff = append(diff, " "+line)
		} else {
			// zip hashes should never change once listed
			t.Errorf("policy violation: zip file hash is changing:\n-%s+%s", line, fixed)
			lines[i] = fixed
			diff = append(diff, "-"+line, "+"+fixed)
		}
	}

	// Add missing lines.
	// Sort keys to avoid non-determinism, but overall file is not sorted.
	// It will end up time-ordered instead.
	for _, name := range slices.Sorted(maps.Keys(want)) {
		line := want[name]
		lines = append(lines, line)
		diff = append(diff, "+"+line)
	}

	// Show diffs or update file.
	fixed := strings.Join(lines, "")
	if fixed != string(sums) {
		if *update && !t.Failed() {
			t.Logf("updating GOROOT/lib/fips140/fips140.sum:\n%s", strings.Join(diff, ""))
			if err := os.WriteFile(file, []byte(fixed), 0666); err != nil {
				t.Fatal(err)
			}
			return
		}
		t.Errorf("GOROOT/lib/fips140/fips140.sum out of date. changes needed:\n%s", strings.Join(diff, ""))
	}
}
