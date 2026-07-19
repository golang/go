// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build compiler_bootstrap

package main

// walkEnum is empty while cmd/cgo is built against bootstrap go/ast, which
// predates enum nodes. Bootstrap inputs cannot contain enum syntax.
func (*File) walkEnum(any, func(*File, any, astContext)) bool { return false }
