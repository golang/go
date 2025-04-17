// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sigchanyzer defines an Analyzer that detects
// misuse of unbuffered signal as argument to signal.Notify.
//
// # Analyzer sigchanyzer
//
// sigchanyzer: check for unbuffered channel of os.Signal
//
// This checker reports call expression of the form
//
//	signal.Notify(c <-chan os.Signal, sig ...os.Signal),
//
// where c is an unbuffered channel, which can be at risk of missing the signal.
package sigchanyzer
