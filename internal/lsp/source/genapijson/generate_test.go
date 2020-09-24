// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"io/ioutil"
	"testing"

	"golang.org/x/tools/internal/testenv"
)

func TestGenerate(t *testing.T) {
	testenv.NeedsGoBuild(t) // This is a lie. We actually need the source code.
	testenv.NeedsGoPackages(t)

	got, err := ioutil.ReadFile("../api_json.go")
	if err != nil {
		t.Fatal(err)
	}
	want, err := generate()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Error("api_json is out of sync. Run `go generate ./internal/lsp/source` from the root of tools.")
	}
}
