// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buildutil_test

import (
	"fmt"
	"go/build"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"golang.org/x/tools/go/buildutil"
)

func testContainingPackageCaseFold(file, want string) error {
	bp, err := buildutil.ContainingPackage(&build.Default, ".", file)
	if err != nil {
		return err
	}
	if got := bp.ImportPath; got != want {
		return fmt.Errorf("ContainingPackage(%q) = %s, want %s", file, got, want)
	}
	return nil
}

func TestContainingPackageCaseFold(t *testing.T) {
	path := filepath.Join(runtime.GOROOT(), `src\fmt\print.go`)
	err := testContainingPackageCaseFold(path, "fmt")
	if err != nil {
		t.Error(err)
	}
	vol := filepath.VolumeName(path)
	if len(vol) != 2 || vol[1] != ':' {
		t.Fatalf("GOROOT path has unexpected volume name: %v", vol)
	}
	rest := path[len(vol):]
	err = testContainingPackageCaseFold(strings.ToUpper(vol)+rest, "fmt")
	if err != nil {
		t.Error(err)
	}
	err = testContainingPackageCaseFold(strings.ToLower(vol)+rest, "fmt")
	if err != nil {
		t.Error(err)
	}
}
