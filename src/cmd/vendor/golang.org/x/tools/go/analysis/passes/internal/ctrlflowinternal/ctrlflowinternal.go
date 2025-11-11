// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ctrlflowinternal exposes internals of ctrlflow.
// It cannot actually depend on symbols from ctrlflow.
package ctrlflowinternal

import "go/types"

// NoReturn exposes the (*ctrlflow.CFGs).NoReturn method to the buildssa analyzer.
//
// You must link [golang.org/x/tools/go/analysis/passes/ctrlflow] into your
// application for it to be non-nil.
var NoReturn = func(cfgs any, fn *types.Func) bool {
	panic("x/tools/go/analysis/passes/ctrlflow is not linked into this application")
}
