// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog_test

import (
	"log/slog"
	"os"
)

func Example_discardHandler() {
	removeTime := func(groups []string, a slog.Attr) slog.Attr {
		if a.Key == slog.TimeKey && len(groups) == 0 {
			return slog.Attr{}
		}
		return a
	}
	// A slog.TextHandler can output log messages.
	logger1 := slog.New(slog.NewTextHandler(
		os.Stdout,
		&slog.HandlerOptions{ReplaceAttr: removeTime},
	))
	logger1.Info("message 1")

	// A slog.DiscardHandler will discard all messages.
	logger2 := slog.New(slog.DiscardHandler)
	logger2.Info("message 2")

	// Output:
	// level=INFO msg="message 1"
}
