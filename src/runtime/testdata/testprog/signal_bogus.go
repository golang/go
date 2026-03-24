// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9

package main

import (
	"os"
	"os/signal"
	"syscall"
)

func init() {
	register("SignalBogus", SignalBogus)
}

// signal.Notify should effectively ignore bogus signal numbers. Never writing
// to the channel, but otherwise allowing Notify/Stop as normal.
//
// This is a regression test for https://go.dev/issue/77076, where bogus
// signals used to make Stop hang if there were no real signals installed.
func SignalBogus() {
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.Signal(0xdead))
	signal.Stop(ch)
	println("OK")
}
