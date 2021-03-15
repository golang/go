// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ((darwin && arm64) || (darwin && amd64 && ios)) && x509omitbundledroots
// +build darwin,arm64 darwin,amd64,ios
// +build x509omitbundledroots

package x509

import (
	"strings"
	"testing"
)

func TestOmitBundledRoots(t *testing.T) {
	cp, err := loadSystemRoots()
	if err == nil {
		t.Fatalf("loadSystemRoots = (pool %p, error %v); want non-nil error", cp, err)
	}
	if !strings.Contains(err.Error(), "root bundling disabled") {
		t.Errorf("unexpected error doesn't mention bundling: %v", err)
	}
}
