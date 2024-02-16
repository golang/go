// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the slog checker.

package slog

import "log/slog"

func SlogTest() {
	slog.Info("msg", "a") // ERROR "call to slog.Info missing a final value"
}
