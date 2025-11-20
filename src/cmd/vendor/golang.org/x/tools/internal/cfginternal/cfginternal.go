// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cfginternal exposes internals of go/cfg.
// It cannot actually depend on symbols from go/cfg.
package cfginternal

// IsNoReturn exposes (*cfg.CFG).noReturn to the ctrlflow analyzer.
// TODO(adonovan): add CFG.NoReturn to the public API.
//
// You must link [golang.org/x/tools/go/cfg] into your application for
// this function to be non-nil.
var IsNoReturn = func(cfg any) bool {
	panic("golang.org/x/tools/go/cfg not linked into application")
}
