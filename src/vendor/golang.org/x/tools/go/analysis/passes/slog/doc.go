// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package slog defines an Analyzer that checks for
// mismatched key-value pairs in log/slog calls.
//
// # Analyzer slog
//
// slog: check for invalid structured logging calls
//
// The slog checker looks for calls to functions from the log/slog
// package that take alternating key-value pairs. It reports calls
// where an argument in a key position is neither a string nor a
// slog.Attr, and where a final key is missing its value.
// For example,it would report
//
//	slog.Warn("message", 11, "k") // slog.Warn arg "11" should be a string or a slog.Attr
//
// and
//
//	slog.Info("message", "k1", v1, "k2") // call to slog.Info missing a final value
package slog
