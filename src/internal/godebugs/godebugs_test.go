// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godebugs_test

import (
	"internal/godebugs"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

func TestAll(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	data, err := os.ReadFile("../../../doc/godebug.md")
	if err != nil {
		if os.IsNotExist(err) && (testenv.Builder() == "" || runtime.GOOS != "linux") {
			t.Skip(err)
		}
		t.Fatal(err)
	}
	doc := string(data)

	incs := incNonDefaults(t)

	last := ""
	for _, info := range godebugs.All {
		if info.Name <= last {
			t.Errorf("All not sorted: %s then %s", last, info.Name)
		}
		last = info.Name

		if info.Package == "" {
			t.Errorf("Name=%s missing Package", info.Name)
		}
		if info.Changed != 0 && info.Old == "" {
			t.Errorf("Name=%s has Changed, missing Old", info.Name)
		}
		if info.Old != "" && info.Changed == 0 {
			t.Errorf("Name=%s has Old, missing Changed", info.Name)
		}
		if !strings.Contains(doc, "`"+info.Name+"`") &&
			!strings.Contains(doc, "`"+info.Name+"=") {
			t.Errorf("Name=%s not documented in doc/godebug.md", info.Name)
		}
		if !info.Opaque && !incs[info.Name] {
			t.Errorf("Name=%s missing IncNonDefault calls; see 'go doc internal/godebug'", info.Name)
		}
	}
}

var incNonDefaultRE = regexp.MustCompile(`([\pL\p{Nd}_]+)\.IncNonDefault\(\)`)

func incNonDefaults(t *testing.T) map[string]bool {
	// Build list of all files importing internal/godebug.
	// Tried a more sophisticated search in go list looking for
	// imports containing "internal/godebug", but that turned
	// up a bug in go list instead. #66218
	out, err := exec.Command("go", "list", "-f={{.Dir}}", "std", "cmd").CombinedOutput()
	if err != nil {
		t.Fatalf("go list: %v\n%s", err, out)
	}

	seen := map[string]bool{}
	for _, dir := range strings.Split(string(out), "\n") {
		if dir == "" {
			continue
		}
		files, err := os.ReadDir(dir)
		if err != nil {
			t.Fatal(err)
		}
		for _, file := range files {
			name := file.Name()
			if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
				continue
			}
			data, err := os.ReadFile(filepath.Join(dir, name))
			if err != nil {
				t.Fatal(err)
			}
			for _, m := range incNonDefaultRE.FindAllSubmatch(data, -1) {
				seen[string(m[1])] = true
			}
		}
	}
	return seen
}
