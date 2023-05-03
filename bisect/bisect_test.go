// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bisect

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// In order for package bisect to be copied into the standard library
// and used by very low-level packages such as internal/godebug,
// it needs to have no imports at all.
func TestNoImports(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		data, err := os.ReadFile(file)
		if err != nil {
			t.Error(err)
			continue
		}
		if strings.Contains(string(data), "\nimport") {
			t.Errorf("%s contains imports; package bisect must not import other packages", file)
		}
	}
}
