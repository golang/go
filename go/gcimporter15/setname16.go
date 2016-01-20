// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.6

package gcimporter

import "go/types"

func setName(pkg *types.Package, name string) {
	pkg.SetName(name)
}
