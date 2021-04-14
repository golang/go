// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The current implementation of notes on Darwin is not async-signal-safe,
// so on Darwin the sigqueue code uses different functions to wake up the
// signal_recv thread. This file holds the non-Darwin implementations of
// those functions. These functions will never be called.

// +build !darwin
// +build !plan9

package runtime

func sigNoteSetup(*note) {
	throw("sigNoteSetup")
}

func sigNoteSleep(*note) {
	throw("sigNoteSleep")
}

func sigNoteWakeup(*note) {
	throw("sigNoteWakeup")
}
