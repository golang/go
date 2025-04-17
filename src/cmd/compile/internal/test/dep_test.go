// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"internal/testenv"
	"strings"
	"testing"
)

func TestDeps(t *testing.T) {
	out, err := testenv.Command(t, testenv.GoToolPath(t), "list", "-f", "{{.Deps}}", "cmd/compile/internal/gc").Output()
	if err != nil {
		t.Fatal(err)
	}
	for _, dep := range strings.Fields(strings.Trim(string(out), "[]")) {
		switch dep {
		case "go/build", "go/scanner":
			// cmd/compile/internal/importer introduces a dependency
			// on go/build and go/token; cmd/compile/internal/ uses
			// go/constant which uses go/token in its API. Once we
			// got rid of those dependencies, enable this check again.
			// TODO(gri) fix this
			// t.Errorf("undesired dependency on %q", dep)
		}
	}
}
