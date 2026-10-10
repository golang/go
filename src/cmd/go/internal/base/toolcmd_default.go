//go:build !windows || !sharedtool

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import "cmd/go/internal/cfg"

// ToolSuffix reports the filename suffix used for builtin tool entry points.
func ToolSuffix() string { return cfg.ToolExeSuffix() }

// ToolCommandLine leaves native tool commands unchanged.
func ToolCommandLine(cmdline []string) ([]string, error) { return cmdline, nil }
