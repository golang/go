// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo

package cgotest

import (
	"testing"

	"cmd/cgo/internal/test/issue9026"
)

func test9026(t *testing.T) { issue9026.Test(t) }
