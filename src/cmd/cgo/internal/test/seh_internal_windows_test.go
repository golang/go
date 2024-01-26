// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo && windows && internal

package cgotest

import (
	"internal/testenv"
	"testing"
)

func TestCallbackCallersSEH(t *testing.T) {
	testenv.SkipFlaky(t, 65116)
}
