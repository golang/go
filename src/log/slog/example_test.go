// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog_test

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"time"
)

func ExampleGroup() {
	r, _ := http.NewRequest("GET", "localhost", nil)
	// ...

	logger := slog.New(
		slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
			ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
				if a.Key == slog.TimeKey && len(groups) == 0 {
					return slog.Attr{}
				}
				return a
			},
		}),
	)
	logger.Info("finished",
		slog.Group("req",
			slog.String("method", r.Method),
			slog.String("url", r.URL.String())),
		slog.Int("status", http.StatusOK),
		slog.Duration("duration", time.Second))

	// Output:
	// level=INFO msg=finished req.method=GET req.url=localhost status=200 duration=1s
}

func ExampleGroupAttrs() {
	r, _ := http.NewRequest("POST", "localhost", http.NoBody)
	// ...

	logger := slog.New(
		slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
			Level: slog.LevelDebug,
			ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
				if a.Key == slog.TimeKey && len(groups) == 0 {
					return slog.Attr{}
				}
				return a
			},
		}),
	)

	// Use []slog.Attr to accumulate attributes.
	attrs := []slog.Attr{slog.String("method", r.Method)}
	attrs = append(attrs, slog.String("url", r.URL.String()))

	if r.Method == "POST" {
		attrs = append(attrs, slog.Int("content-length", int(r.ContentLength)))
	}

	// Group the attributes under a key.
	logger.LogAttrs(context.Background(), slog.LevelInfo,
		"finished",
		slog.Int("status", http.StatusOK),
		slog.GroupAttrs("req", attrs...),
	)

	// Groups with empty keys are inlined.
	logger.LogAttrs(context.Background(), slog.LevelInfo,
		"finished",
		slog.Int("status", http.StatusOK),
		slog.GroupAttrs("", attrs...),
	)

	// Output:
	// level=INFO msg=finished status=200 req.method=POST req.url=localhost req.content-length=0
	// level=INFO msg=finished status=200 method=POST url=localhost content-length=0
}
