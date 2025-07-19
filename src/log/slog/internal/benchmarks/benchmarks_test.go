// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package benchmarks

import (
	"context"
	"flag"
	"internal/race"
	"io"
	"log/slog"
	"log/slog/internal"
	"testing"
)

func init() {
	flag.BoolVar(&internal.IgnorePC, "nopc", false, "do not invoke runtime.Callers")
}

// We pass Attrs inline because it affects allocations: building
// up a list outside of the benchmarked code and passing it in with "..."
// reduces measured allocations.

func BenchmarkAttrs(b *testing.B) {
	ctx := context.Background()
	for _, handler := range []struct {
		name     string
		h        slog.Handler
		skipRace bool
	}{
		{"disabled", disabledHandler{}, false},
		{"async discard", newAsyncHandler(), true},
		{"fastText discard", newFastTextHandler(io.Discard), false},
		{"Text discard", slog.NewTextHandler(io.Discard, nil), false},
		{"JSON discard", slog.NewJSONHandler(io.Discard, nil), false},
	} {
		logger := slog.New(handler.h)
		b.Run(handler.name, func(b *testing.B) {
			if handler.skipRace && race.Enabled {
				b.Skip("skipping benchmark in race mode")
			}
			for _, call := range []struct {
				name string
				f    func()
			}{
				{
					// The number should match nAttrsInline in slog/record.go.
					// This should exercise the code path where no allocations
					// happen in Record or Attr. If there are allocations, they
					// should only be from Duration.String and Time.String.
					"5 args",
					func() {
						logger.LogAttrs(nil, slog.LevelInfo, testMessage,
							slog.String("string", testString),
							slog.Int("status", testInt),
							slog.Duration("duration", testDuration),
							slog.Time("time", testTime),
							slog.Any("error", testError),
						)
					},
				},
				{
					"5 args ctx",
					func() {
						logger.LogAttrs(ctx, slog.LevelInfo, testMessage,
							slog.String("string", testString),
							slog.Int("status", testInt),
							slog.Duration("duration", testDuration),
							slog.Time("time", testTime),
							slog.Any("error", testError),
						)
					},
				},
				{
					"10 args",
					func() {
						logger.LogAttrs(nil, slog.LevelInfo, testMessage,
							slog.String("string", testString),
							slog.Int("status", testInt),
							slog.Duration("duration", testDuration),
							slog.Time("time", testTime),
							slog.Any("error", testError),
							slog.String("string", testString),
							slog.Int("status", testInt),
							slog.Duration("duration", testDuration),
							slog.Time("time", testTime),
							slog.Any("event", testEvent),
						)
					},
				},
				{
					// Try an extreme value to see if the results are reasonable.
					"40 args",
					func() {
						logger.LogAttrs(nil, slog.LevelInfo, testMessage,
							slog.String("string", testString),
							slog.Int("status", testInt),
							slog.Duration("duration", testDuration),
							slog.Time("time", testTime),
							slog.Any("error", testError),
							slog.String("string", testString),
							slog.Int("status", testInt),
							slog.Duration("duration", testDuration),
							slog.Time("time", testTime),
							slog.Any("error", testError),
							slog.String("string", testString),
							slog.Int("status", testInt),
							slog.Duration("duration", testDuration),
							slog.Time("time", testTime),
							slog.Any("error", testError),
							slog.String("string", testString),
							slog.Int("status", testInt),
							slog.Duration("duration", testDuration),
							slog.Time("time", testTime),
							slog.Any("error", testError),
							slog.String("string", testString),
							slog.Int("status", testInt),
							slog.Duration("duration", testDuration),
							slog.Time("time", testTime),
							slog.Any("error", testError),
							slog.String("string", testString),
							slog.Int("status", testInt),
							slog.Duration("duration", testDuration),
							slog.Time("time", testTime),
							slog.Any("error", testError),
							slog.String("string", testString),
							slog.Int("status", testInt),
							slog.Duration("duration", testDuration),
							slog.Time("time", testTime),
							slog.Any("error", testError),
							slog.String("string", testString),
							slog.Int("status", testInt),
							slog.Duration("duration", testDuration),
							slog.Time("time", testTime),
							slog.Any("event", testEvent),
						)
					},
				},
			} {
				b.Run(call.name, func(b *testing.B) {
					b.ReportAllocs()
					b.RunParallel(func(pb *testing.PB) {
						for pb.Next() {
							call.f()
						}
					})
				})
			}
		})
	}
}
