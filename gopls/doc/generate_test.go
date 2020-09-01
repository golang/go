// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"testing"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/testenv"
)

func TestGenerated(t *testing.T) {
	testenv.NeedsGoBuild(t) // This is a lie. We actually need the source code.

	var opts map[string][]option
	if err := json.Unmarshal([]byte(source.OptionsJson), &opts); err != nil {
		t.Fatal(err)
	}

	doc, err := ioutil.ReadFile("settings.md")
	if err != nil {
		t.Fatal(err)
	}

	got, err := rewriteDoc(doc, opts)
	if !bytes.Equal(got, doc) {
		t.Error("settings.md needs updating. run: `go run gopls/doc/generate.go` from the root of tools.")
	}
}
