// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog_test

import (
	"bytes"
	"log/slog"
	"os"
)

func ExampleMultiHandler() {
	removeTime := func(groups []string, a slog.Attr) slog.Attr {
		if a.Key == slog.TimeKey && len(groups) == 0 {
			return slog.Attr{}
		}
		return a
	}

	var textBuf, jsonBuf bytes.Buffer
	textHandler := slog.NewTextHandler(&textBuf, &slog.HandlerOptions{ReplaceAttr: removeTime})
	jsonHandler := slog.NewJSONHandler(&jsonBuf, &slog.HandlerOptions{ReplaceAttr: removeTime})

	multiHandler := slog.MultiHandler(textHandler, jsonHandler)
	logger := slog.New(multiHandler)

	logger.Info("login",
		slog.String("name", "whoami"),
		slog.Int("id", 42),
	)

	os.Stdout.WriteString(textBuf.String())
	os.Stdout.WriteString(jsonBuf.String())

	// Output:
	// level=INFO msg=login name=whoami id=42
	// {"level":"INFO","msg":"login","name":"whoami","id":42}
}
