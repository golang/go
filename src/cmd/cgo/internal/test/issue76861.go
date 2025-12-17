// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo

package cgotest

// Issue 43639: No runtime test needed, make sure package
// cmd/cgo/internal/test/issue76861 compiles without error.

import _ "cmd/cgo/internal/test/issue76861"
