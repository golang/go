// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo

package cgotest

import (
	"testing"

	"cmd/cgo/internal/test/issue26213"
)

func test26213(t *testing.T) {
	issue26213.Test26213(t)
}
