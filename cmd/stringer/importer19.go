// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.9

package main

import (
	"go/importer"
	"go/types"
)

func defaultImporter() types.Importer {
	return importer.For("source", nil)
}
