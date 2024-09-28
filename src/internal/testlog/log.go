// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package testlog provides a back-channel communication path
// between tests and package os, so that cmd/go can see which
// environment variables and files a test consults.
package testlog

import "sync/atomic"

// Interface is the interface required of test loggers.
// The os package will invoke the interface's methods to indicate that
// it is inspecting the given environment variables or files.
// Multiple goroutines may call these methods simultaneously.
type Interface interface {
	Getenv(key string)
	Stat(file string)
	Open(file string)
	Chdir(dir string)
}

// logger is the current logger Interface.
// We use an atomic.Value in case test startup
// is racing with goroutines started during init.
// That must not cause a race detector failure,
// although it will still result in limited visibility
// into exactly what those goroutines do.
var logger atomic.Pointer[Interface]

// SetLogger sets the test logger implementation for the current process.
// It must be called only once, at process startup.
func SetLogger(impl Interface) {
	if !logger.CompareAndSwap(nil, &impl) {
		panic("testlog: SetLogger must be called only once")
	}
}

// Logger returns the current test logger implementation.
// It returns nil if there is no logger.
func Logger() Interface {
	impl := logger.Load()
	if impl == nil {
		return nil
	}
	return *impl
}

// Getenv calls Logger().Getenv, if a logger has been set.
func Getenv(name string) {
	if log := Logger(); log != nil {
		log.Getenv(name)
	}
}

// Open calls Logger().Open, if a logger has been set.
func Open(name string) {
	if log := Logger(); log != nil {
		log.Open(name)
	}
}

// Stat calls Logger().Stat, if a logger has been set.
func Stat(name string) {
	if log := Logger(); log != nil {
		log.Stat(name)
	}
}
