// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"testing"

	"golang.org/x/tools/internal/testenv"
)

func TestGenerated(t *testing.T) {
	testenv.NeedsGoPackages(t)
	// This test fails on Kokoro, for unknown reasons, so must be run only on TryBots.
	// In any case, it suffices to run this test on any builder.
	testenv.NeedsGo1Point(t, 21)

	testenv.NeedsLocalXTools(t)

	ok, err := doMain(false)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Error("documentation needs updating. Run: cd gopls && go generate")
	}
}
