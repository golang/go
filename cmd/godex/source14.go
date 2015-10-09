// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.5

// This file implements access to export data from source.

package main

import (
	"golang.org/x/tools/go/types"
)

func init() {
	register("source", sourceImporter)
}

func sourceImporter(packages map[string]*types.Package, path string) (*types.Package, error) {
	panic("unimplemented")
}
