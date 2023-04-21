// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the slog checker.

//go:build go1.21

package b

import "a"

func Imported() {
	_ = a.MyLogger.With("a", 1, 2, 3) // want `slog.Logger.With arg "2" should be a string or a slog.Attr`
}
