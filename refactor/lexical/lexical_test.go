// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Incomplete std lib sources on Android.

// +build !android

package lexical

import (
	"go/build"
	"testing"

	"golang.org/x/tools/go/buildutil"
	"golang.org/x/tools/go/loader"
)

func TestStdlib(t *testing.T) {
	defer func(saved func(format string, args ...interface{})) {
		logf = saved
	}(logf)
	logf = t.Errorf

	ctxt := build.Default // copy

	// Enumerate $GOROOT packages.
	saved := ctxt.GOPATH
	ctxt.GOPATH = "" // disable GOPATH during AllPackages
	pkgs := buildutil.AllPackages(&ctxt)
	ctxt.GOPATH = saved

	// Throw in a number of go.tools packages too.
	pkgs = append(pkgs,
		"golang.org/x/tools/cmd/godoc",
		"golang.org/x/tools/refactor/lexical")

	// Load, parse and type-check the program.
	conf := loader.Config{Build: &ctxt}
	for _, path := range pkgs {
		conf.ImportWithTests(path)
	}
	iprog, err := conf.Load()
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// This test ensures that Structure doesn't panic and that
	// its internal sanity-checks against go/types don't fail.
	for pkg, info := range iprog.AllPackages {
		_ = Structure(iprog.Fset, pkg, &info.Info, info.Files)
	}
}
