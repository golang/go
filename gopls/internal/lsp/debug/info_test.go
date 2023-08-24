// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package debug exports debug information for gopls.
package debug

import (
	"bytes"
	"context"
	"encoding/json"
	"runtime"
	"testing"
)

func TestPrintVersionInfoJSON(t *testing.T) {
	buf := new(bytes.Buffer)
	if err := PrintVersionInfo(context.Background(), buf, true, JSON); err != nil {
		t.Fatalf("PrintVersionInfo failed: %v", err)
	}
	res := buf.Bytes()

	var got ServerVersion
	if err := json.Unmarshal(res, &got); err != nil {
		t.Fatalf("unexpected output: %v\n%s", err, res)
	}
	if g, w := got.GoVersion, runtime.Version(); g != w {
		t.Errorf("go version = %v, want %v", g, w)
	}
	if g, w := got.Version, Version(); g != w {
		t.Errorf("gopls version = %v, want %v", g, w)
	}
	// Other fields of BuildInfo may not be available during test.
}

func TestPrintVersionInfoPlainText(t *testing.T) {
	buf := new(bytes.Buffer)
	if err := PrintVersionInfo(context.Background(), buf, true, PlainText); err != nil {
		t.Fatalf("PrintVersionInfo failed: %v", err)
	}
	res := buf.Bytes()

	// Other fields of BuildInfo may not be available during test.
	wantGoplsVersion, wantGoVersion := Version(), runtime.Version()
	if !bytes.Contains(res, []byte(wantGoplsVersion)) || !bytes.Contains(res, []byte(wantGoVersion)) {
		t.Errorf("plaintext output = %q,\nwant (version: %v, go: %v)", res, wantGoplsVersion, wantGoVersion)
	}
}
