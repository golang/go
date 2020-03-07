// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package log is a context based logging package, designed to interact well
// with both the lsp protocol and the other telemetry packages.
package log

import "golang.org/x/tools/internal/telemetry/event"

var (
	With  = event.Log
	Print = event.Print
	Error = event.Error
)
