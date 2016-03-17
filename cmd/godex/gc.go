// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.5

// This file implements access to gc-generated export data.

package main

import "go/importer"

func init() {
	register("gc", importer.For("gc", nil))
}
