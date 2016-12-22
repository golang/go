// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements access to export data from source.

package main

import "go/types"

func init() {
	register("source", sourceImporter{})
}

type sourceImporter struct{}

func (sourceImporter) Import(path string) (*types.Package, error) {
	panic("unimplemented")
}
