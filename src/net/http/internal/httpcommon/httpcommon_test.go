// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httpcommon_test

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// This package is imported by the net/http package,
// and therefore must not itself import net/http.
func TestNoNetHttp(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}
		// Could use something complex like go/build or x/tools/go/packages,
		// but there's no reason for "net/http" to appear (in quotes) in the source
		// otherwise, so just use a simple substring search.
		data, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(data, []byte(`"net/http"`)) {
			t.Errorf(`%s: cannot import "net/http"`, file)
		}
	}
}
