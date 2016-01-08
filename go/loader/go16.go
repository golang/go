// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.6

package loader

import "go/build"

func init() {
	ignoreVendor = build.IgnoreVendor
}
