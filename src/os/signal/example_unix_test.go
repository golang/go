// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package signal_test

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
)

var neverReady = make(chan struct{}) // never closed

// This example passes a context with a signal to tell a blocking function that
// it should abandon its work after a signal is received.
func ExampleNotifyContext() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	p, err := os.FindProcess(os.Getpid())
	if err != nil {
		log.Fatal(err)
	}

	// On a Unix-like system, pressing Ctrl+C on a keyboard sends a
	// SIGINT signal to the process of the program in execution.
	//
	// This example simulates that by sending a SIGINT signal to itself.
	if err := p.Signal(os.Interrupt); err != nil {
		log.Fatal(err)
	}

	select {
	case <-neverReady:
		fmt.Println("ready")
	case <-ctx.Done():
		fmt.Println(ctx.Err()) // prints "context canceled"
		stop()                 // stop receiving signal notifications as soon as possible.
	}

	// Output:
	// context canceled
}
