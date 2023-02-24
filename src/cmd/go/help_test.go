// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"bytes"
	"go/format"
	diffpkg "internal/diff"
	"os"
	"testing"

	"cmd/go/internal/help"
	"cmd/go/internal/modload"
)

func TestDocsUpToDate(t *testing.T) {
	t.Parallel()

	if !modload.Enabled() {
		t.Skipf("help.Help in GOPATH mode is configured by main.main")
	}

	buf := new(bytes.Buffer)
	// Match the command in mkalldocs.sh that generates alldocs.go.
	help.Help(buf, []string{"documentation"})
	internal := buf.Bytes()
	internal, err := format.Source(internal)
	if err != nil {
		t.Fatalf("gofmt docs: %v", err)
	}
	alldocs, err := os.ReadFile("alldocs.go")
	if err != nil {
		t.Fatalf("error reading alldocs.go: %v", err)
	}
	if !bytes.Equal(internal, alldocs) {
		t.Errorf("alldocs.go is not up to date; run mkalldocs.sh to regenerate it\n%s",
			diffpkg.Diff("go help documentation | gofmt", internal, "alldocs.go", alldocs))
	}
}
