// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import "testing"

// The actual test functions are in non-_test.go files
// so that they can use cgo (import "C").
// These wrappers are here for gotest to find.

func Test8756(t *testing.T)     { test8756(t) }
func Test9026(t *testing.T)     { test9026(t) }
func Test9510(t *testing.T)     { test9510(t) }
func Test20266(t *testing.T)    { test20266(t) }
func Test26213(t *testing.T)    { test26213(t) }
func TestGCC68255(t *testing.T) { testGCC68255(t) }
