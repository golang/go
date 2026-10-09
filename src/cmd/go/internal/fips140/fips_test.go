// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips140

import (
	"archive/zip"
	"cmp"
	"crypto/sha256"
	"flag"
	"fmt"
	"internal/testenv"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"cmd/go/internal/cfg"

	"golang.org/x/mod/sumdb/dirhash"
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

	format := func(name string, sum [32]byte, ziphash string) string {
		return fmt.Sprintf("%s %x %s\n", name, sum[:], ziphash)
	}

	want := make(map[string]string)
	for _, zip := range zips {
		data, err := os.ReadFile(zip)
		if err != nil {
			t.Fatal(err)
		}
		ziphash, err := dirhash.HashZip(zip, dirhash.DefaultHash)
		if err != nil {
			t.Fatal(err)
		}
		name := filepath.Base(zip)
		want[name] = format(name, sha256.Sum256(data), ziphash)
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

// TestZipCompression checks that files with the same contents in different
// snapshot zip files are compressed to the same bytes, so that git can store
// each new zip file as a small delta against the previous ones.
func TestZipCompression(t *testing.T) {
	lib := filepath.Join(testenv.GOROOT(t), "lib/fips140")
	files, err := filepath.Glob(filepath.Join(lib, "*.zip"))
	if err != nil {
		t.Fatal(err)
	}

	type file struct {
		name string
		sum  [32]byte
	}
	type compressed struct {
		zip string
		sum [32]byte
	}
	// (trimmed path, uncompressed hash) => (zip file name, compressed hash)
	seen := make(map[file]compressed)
	// (zip file name A, zip file name B) => number of mismatching files
	mismatches := make(map[[2]string]int)
	for _, path := range files {
		name := filepath.Base(path)
		if name == "v1.0.1.zip" {
			// Compressed by the Go 1.27 encoder before mkzip required Go 1.26.
			continue
		}
		z, err := zip.OpenReader(path)
		if err != nil {
			t.Fatal(err)
		}
		for _, f := range z.File {
			rc, err := f.Open()
			if err != nil {
				t.Fatal(err)
			}
			contents, err := io.ReadAll(rc)
			rc.Close()
			if err != nil {
				t.Fatal(err)
			}
			r, err := f.OpenRaw()
			if err != nil {
				t.Fatal(err)
			}
			raw, err := io.ReadAll(r)
			if err != nil {
				t.Fatal(err)
			}
			// golang.org/fips140@VERSION/fips140/VERSION/....
			parts := strings.SplitN(f.Name, "/", 5)
			k := file{parts[len(parts)-1], sha256.Sum256(contents)}
			c := compressed{name, sha256.Sum256(raw)}
			if prev, ok := seen[k]; ok {
				if prev.sum != c.sum {
					mismatches[[2]string{prev.zip, name}]++
				}
				continue
			}
			seen[k] = c
		}
		z.Close()
	}

	for _, pair := range slices.SortedFunc(maps.Keys(mismatches), func(a, b [2]string) int {
		return cmp.Or(strings.Compare(a[0], b[0]), strings.Compare(a[1], b[1]))
	}) {
		t.Errorf("%d files with the same contents in %s and %s are compressed differently",
			mismatches[pair], pair[0], pair[1])
	}
}

func TestVerifyZipSum(t *testing.T) {
	dir := t.TempDir()
	zipfile := filepath.Join(dir, "v1.2.3.zip")
	data := []byte("not really a zip, but it does not matter here")
	if err := os.WriteFile(zipfile, data, 0666); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(data)

	sumfile := filepath.Join(dir, "fips140.sum")
	write := func(contents string) {
		if err := os.WriteFile(sumfile, []byte(contents), 0666); err != nil {
			t.Fatal(err)
		}
	}

	const (
		zeroSum = "0000000000000000000000000000000000000000000000000000000000000000"
		ziphash = "h1:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
	)

	// Matching hash with comments and a second entry passes.
	write(fmt.Sprintf("# comment\n\nv1.2.3.zip %x %s\nother.zip %s h1:other=\n", sum[:], ziphash, zeroSum))
	if err := verifyZipSum(zipfile, sumfile); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Missing entry for this zip fails.
	write("# only comments\n")
	if err := verifyZipSum(zipfile, sumfile); err == nil {
		t.Errorf("expected error when hash entry is missing")
	}

	// Wrong hash fails.
	write(fmt.Sprintf("v1.2.3.zip %s %s\n", zeroSum, ziphash))
	if err := verifyZipSum(zipfile, sumfile); err == nil {
		t.Errorf("expected error when hash does not match")
	}

	// Missing sum file fails.
	if err := verifyZipSum(zipfile, filepath.Join(dir, "does-not-exist.sum")); err == nil {
		t.Errorf("expected error when sum file is missing")
	}
}

func TestAliases(t *testing.T) {
	lib := filepath.Join(testenv.GOROOT(t), "lib/fips140")
	txts, err := filepath.Glob(filepath.Join(lib, "*.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if len(txts) == 0 {
		t.Fatal("no *.txt files in GOROOT/lib/fips140")
	}

	defer func(origGOFIPS140, origVersion string, origTags []string) {
		cfg.GOFIPS140 = origGOFIPS140
		cfg.BuildContext.BuildTags = origTags
		version = origVersion
	}(cfg.GOFIPS140, version, cfg.BuildContext.BuildTags)

	for _, txt := range txts {
		alias := strings.TrimSuffix(filepath.Base(txt), ".txt")
		data, err := os.ReadFile(txt)
		if err != nil {
			t.Fatal(err)
		}
		want := strings.TrimSpace(string(data))
		if want == "" {
			t.Errorf("%s is empty", txt)
			continue
		}
		cfg.GOFIPS140 = alias
		cfg.BuildContext.BuildTags = nil
		version = ""
		initVersion()
		if version != want {
			t.Errorf("initVersion(%q): version = %q, want %q", alias, version, want)
		}
	}
}
