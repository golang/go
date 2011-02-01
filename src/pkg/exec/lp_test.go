// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

import (
	"testing"
)

var nonExistentPaths = []string{
	"some-non-existent-path",
	"non-existent-path/slashed",
}

func TestLookPathNotFound(t *testing.T) {
	for _, name := range nonExistentPaths {
		path, err := LookPath(name)
		if err == nil {
			t.Fatalf("LookPath found %q in $PATH", name)
		}
		if path != "" {
			t.Fatalf("LookPath path == %q when err != nil", path)
		}
		perr, ok := err.(*PathError)
		if !ok {
			t.Fatal("LookPath error is not a PathError")
		}
		if perr.Name != name {
			t.Fatalf("want PathError name %q, got %q", name, perr.Name)
		}
	}
}
