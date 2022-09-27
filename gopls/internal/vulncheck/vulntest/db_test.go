// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package vulntest

import (
	"context"
	"encoding/json"
	"testing"
)

func TestNewDatabase(t *testing.T) {
	ctx := context.Background()
	in := []byte(`
-- GO-2020-0001.yaml --
modules:
  - module: github.com/gin-gonic/gin
    versions:
      - fixed: 1.6.0
    packages:
      - package: github.com/gin-gonic/gin
        symbols:
          - defaultLogFormatter
description: |
    Something.
published: 2021-04-14T20:04:52Z
references:
  - fix: https://github.com/gin-gonic/gin/pull/2237
`)

	db, err := NewDatabase(ctx, in)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Clean()

	cli, err := NewClient(db)
	if err != nil {
		t.Fatal(err)
	}
	got, err := cli.GetByID(ctx, "GO-2020-0001")
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != "GO-2020-0001" {
		m, _ := json.Marshal(got)
		t.Errorf("got %s\nwant GO-2020-0001 entry", m)
	}
	gotAll, err := cli.GetByModule(ctx, "github.com/gin-gonic/gin")
	if err != nil {
		t.Fatal(err)
	}
	if len(gotAll) != 1 || gotAll[0].ID != "GO-2020-0001" {
		m, _ := json.Marshal(got)
		t.Errorf("got %s\nwant GO-2020-0001 entry", m)
	}
}
