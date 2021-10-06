// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the buildtag checker.

//go:build !(bad || worse)

package testdata

// +build other // ERROR "misplaced \+build comment"
