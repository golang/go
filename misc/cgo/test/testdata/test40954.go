// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import (
	"testing"

	"cgotest/issue40954"
)

func test40954(t *testing.T) {
	issue40954.Test40954(t)
}
