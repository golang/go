// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.5

package gcimporter

import (
	"go/types"
	"testing"
)

func TestSetName(t *testing.T) {
	pkg := types.NewPackage("path", "foo")
	scope := pkg.Scope()

	// verify setName
	setName(pkg, "bar")
	if name := pkg.Name(); name != "bar" {
		t.Fatalf(`got package name %q; want "bar"`, name)
	}

	// verify no other fields are changed
	if pkg.Path() != "path" || pkg.Scope() != scope || pkg.Complete() || pkg.Imports() != nil {
		t.Fatalf("setName changed other fields")
	}
}
