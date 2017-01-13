// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo

package main

import (
	"testing"
)

func TestInternalLinkerCgoFile(t *testing.T) {
	testGoFile(t, true, false)
}

func TestExternalLinkerCgoFile(t *testing.T) {
	testGoFile(t, true, true)
}
