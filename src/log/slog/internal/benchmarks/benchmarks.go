// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package benchmarks contains benchmarks for slog.
//
// These benchmarks are loosely based on github.com/uber-go/zap/benchmarks.
// They have the following desirable properties:
//
//   - They test a complete log event, from the user's call to its return.
//
//   - The benchmarked code is run concurrently in multiple goroutines, to
//     better simulate a real server (the most common environment for structured
//     logs).
//
//   - Some handlers are optimistic versions of real handlers, doing real-world
//     tasks as fast as possible (and sometimes faster, in that an
//     implementation may not be concurrency-safe). This gives us an upper bound
//     on handler performance, so we can evaluate the (handler-independent) core
//     activity of the package in an end-to-end context without concern that a
//     slow handler implementation is skewing the results.
//
//   - We also test the built-in handlers, for comparison.
package benchmarks

import (
	"errors"
	"log/slog"
	"time"
)

const testMessage = "Test logging, but use a somewhat realistic message length."

var (
	testTime     = time.Date(2022, time.May, 1, 0, 0, 0, 0, time.UTC)
	testString   = "7e3b3b2aaeff56a7108fe11e154200dd/7819479873059528190"
	testInt      = 32768
	testDuration = 23 * time.Second
	testError    = errors.New("fail")
)

var testAttrs = []slog.Attr{
	slog.String("string", testString),
	slog.Int("status", testInt),
	slog.Duration("duration", testDuration),
	slog.Time("time", testTime),
	slog.Any("error", testError),
}

const wantText = "time=1651363200 level=0 msg=Test logging, but use a somewhat realistic message length. string=7e3b3b2aaeff56a7108fe11e154200dd/7819479873059528190 status=32768 duration=23000000000 time=1651363200 error=fail\n"
