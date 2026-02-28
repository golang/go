// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"flag"
	"go/format"
	"internal/diff"
	"internal/testenv"
	"os"
	"strings"
	"testing"
)

var fixDocs = flag.Bool("fixdocs", false, "if true, update alldocs.go")

func TestDocsUpToDate(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	if !*fixDocs {
		t.Parallel()
	}

	// We run 'go help documentation' as a subprocess instead of
	// calling help.Help directly because it may be sensitive to
	// init-time configuration
	cmd := testenv.Command(t, testGo, "help", "documentation")
	// Unset GO111MODULE so that the 'go get' section matches
	// the default 'go get' implementation.
	cmd.Env = append(cmd.Environ(), "GO111MODULE=")
	cmd.Stderr = new(strings.Builder)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("%v: %v\n%s", cmd, err, cmd.Stderr)
	}

	alldocs, err := format.Source(out)
	if err != nil {
		t.Fatalf("format.Source($(%v)): %v", cmd, err)
	}

	const srcPath = `alldocs.go`
	old, err := os.ReadFile(srcPath)
	if err != nil {
		t.Fatalf("error reading %s: %v", srcPath, err)
	}
	diff := diff.Diff(srcPath, old, "go help documentation | gofmt", alldocs)
	if diff == nil {
		t.Logf("%s is up to date.", srcPath)
		return
	}

	if *fixDocs {
		if err := os.WriteFile(srcPath, alldocs, 0666); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %d bytes to %s", len(alldocs), srcPath)
	} else {
		t.Logf("\n%s", diff)
		t.Errorf("%s is stale. To update, run 'go generate cmd/go'.", srcPath)
	}
}
