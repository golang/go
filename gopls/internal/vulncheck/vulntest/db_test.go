// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package vulntest

import (
	"context"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/gopls/internal/vulncheck/osv"
)

var update = flag.Bool("update", false, "update golden files in testdata/")

func TestNewDatabase(t *testing.T) {
	ctx := context.Background()

	in, err := os.ReadFile("testdata/report.yaml")
	if err != nil {
		t.Fatal(err)
	}
	in = append([]byte("-- GO-2020-0001.yaml --\n"), in...)

	db, err := NewDatabase(ctx, in)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Clean()
	dbpath := span.URIFromURI(db.URI()).Filename()

	// The generated JSON file will be in DB/GO-2022-0001.json.
	got := readOSVEntry(t, filepath.Join(dbpath, "GO-2020-0001.json"))
	got.Modified = time.Time{}

	if *update {
		updateTestData(t, got, "testdata/GO-2020-0001.json")
	}

	want := readOSVEntry(t, "testdata/GO-2020-0001.json")
	want.Modified = time.Time{}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("mismatch (-want +got):\n%s", diff)
	}
}

func updateTestData(t *testing.T, got *osv.Entry, fname string) {
	content, err := json.MarshalIndent(got, "", "\t")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fname, content, 0666); err != nil {
		t.Fatal(err)
	}
	t.Logf("updated %v", fname)
}

func readOSVEntry(t *testing.T, filename string) *osv.Entry {
	t.Helper()
	content, err := os.ReadFile(filename)
	if err != nil {
		t.Fatal(err)
	}
	var entry osv.Entry
	if err := json.Unmarshal(content, &entry); err != nil {
		t.Fatal(err)
	}
	return &entry
}
