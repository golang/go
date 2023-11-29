// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flag_test

import (
	"flag"
	"fmt"
	"os"
	"time"
)

func ExampleFlagSet() {
	start := func(args []string) {
		fs := flag.NewFlagSet("start", flag.ExitOnError)
		addr := fs.String("address", ":8080", "address to listen on")
		fs.Parse(args)
		fmt.Printf("starting server on %s\n", *addr)
	}

	stop := func(args []string) {
		fs := flag.NewFlagSet("stop", flag.ExitOnError)
		timeout := fs.Duration("timeout", time.Second, "stop timeout")
		fs.Parse(args)
		fmt.Printf("stopping server (timeout=%v)\n", *timeout)
	}

	main := func(args []string) {
		subArgs := args[2:] // Drop program name and command.
		switch args[1] {
		case "start":
			start(subArgs)
		case "stop":
			stop(subArgs)
		default:
			fmt.Fprintf(os.Stderr, "error: unknown command - %q\n", args[1])
			os.Exit(1)
		}
	}

	main([]string{"httpd", "start", "-address", ":9999"})
	main([]string{"httpd", "stop"})

	// Output:
	// starting server on :9999
	// stopping server (timeout=1s)
}
