// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"

	"internal/trace"
	"internal/trace/testtrace"
)

func init() {
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage: %s\n", os.Args[0])
		fmt.Fprintf(flag.CommandLine.Output(), "\n")
		fmt.Fprintf(flag.CommandLine.Output(), "Accepts a trace at stdin and validates it.\n")
		flag.PrintDefaults()
	}
	log.SetFlags(0)
}

var logEvents = flag.Bool("log-events", false, "whether to log events")

func main() {
	flag.Parse()

	r, err := trace.NewReader(os.Stdin)
	if err != nil {
		log.Fatal(err)
	}
	v := testtrace.NewValidator()
	for {
		ev, err := r.ReadEvent()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		if *logEvents {
			log.Println(ev.String())
		}
		if err := v.Event(ev); err != nil {
			log.Fatal(err)
		}
	}
}
