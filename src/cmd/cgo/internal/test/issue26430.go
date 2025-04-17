// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo

// Issue 26430: incomplete typedef leads to inconsistent typedefs error.
// No runtime test; just make sure it compiles.

package cgotest

import _ "cmd/cgo/internal/test/issue26430"
