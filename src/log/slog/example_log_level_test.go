// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog_test

import (
	"log"
	"log/slog"
	"log/slog/internal/slogtest"
	"os"
)

// This example shows how to use slog.SetLogLoggerLevel to change the minimal level
// of the internal default handler for slog package before calling slog.SetDefault.
func ExampleSetLogLoggerLevel_log() {
	defer log.SetFlags(log.Flags()) // revert changes after the example
	log.SetFlags(0)
	defer log.SetOutput(log.Writer()) // revert changes after the example
	log.SetOutput(os.Stdout)

	// Default logging level is slog.LevelInfo.
	log.Print("log debug") // log debug
	slog.Debug("debug")    // no output
	slog.Info("info")      // INFO info

	// Set the default logging level to slog.LevelDebug.
	currentLogLevel := slog.SetLogLoggerLevel(slog.LevelDebug)
	defer slog.SetLogLoggerLevel(currentLogLevel) // revert changes after the example

	log.Print("log debug") // log debug
	slog.Debug("debug")    // DEBUG debug
	slog.Info("info")      // INFO info

	// Output:
	// log debug
	// INFO info
	// log debug
	// DEBUG debug
	// INFO info
}

// This example shows how to use slog.SetLogLoggerLevel to change the minimal level
// of the internal writer that uses the custom handler for log package after
// calling slog.SetDefault.
func ExampleSetLogLoggerLevel_slog() {
	// Set the default logging level to slog.LevelError.
	currentLogLevel := slog.SetLogLoggerLevel(slog.LevelError)
	defer slog.SetLogLoggerLevel(currentLogLevel) // revert changes after the example

	defer slog.SetDefault(slog.Default()) // revert changes after the example
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{ReplaceAttr: slogtest.RemoveTime})))

	log.Print("error") // level=ERROR msg=error

	// Output:
	// level=ERROR msg=error
}
