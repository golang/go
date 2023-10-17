// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typesinternal

import "go/types"

// This file contains back doors that allow gopls to avoid method sorting when
// using the objectpath package.
//
// This is performance-critical in certain repositories, but changing the
// behavior of the objectpath package is still being discussed in
// golang/go#61443. If we decide to remove the sorting in objectpath we can
// simply delete these back doors. Otherwise, we should add a new API to
// objectpath that allows controlling the sorting.

// SkipEncoderMethodSorting marks enc (which must be an *objectpath.Encoder) as
// not requiring sorted methods.
var SkipEncoderMethodSorting func(enc interface{})

// ObjectpathObject is like objectpath.Object, but allows suppressing method
// sorting.
var ObjectpathObject func(pkg *types.Package, p string, skipMethodSorting bool) (types.Object, error)
