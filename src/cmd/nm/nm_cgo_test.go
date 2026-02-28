// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"internal/testenv"
	"testing"
)

func TestInternalLinkerCgoExec(t *testing.T) {
	testenv.MustHaveCGO(t)
	// N.B. the go build explictly doesn't pass through
	// -asan/-msan/-race, so we don't care about those.
	testenv.MustInternalLink(t, testenv.SpecialBuildTypes{Cgo: true})
	testGoExec(t, true, false)
}

func TestExternalLinkerCgoExec(t *testing.T) {
	testenv.MustHaveCGO(t)
	testGoExec(t, true, true)
}

func TestCgoLib(t *testing.T) {
	testenv.MustHaveCGO(t)
	testGoLib(t, true)
}
