// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.6

package buildutil

import "go/build"

// AllowVendor is a synonym for zero.
// It allows applications to refer to the go/build.AllowVendor
// feature whether or not it is supported.
const AllowVendor build.ImportMode = 0
