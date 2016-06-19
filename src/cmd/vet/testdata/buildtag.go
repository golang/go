// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the buildtag checker.

// +builder // ERROR "possible malformed \+build comment"
// +build !ignore

package testdata

// +build toolate // ERROR "build comment must appear before package clause and be followed by a blank line"

var _ = 3
