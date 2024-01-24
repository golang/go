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

	"internal/trace/v2/raw"
	"internal/trace/v2/version"
)

func init() {
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage: %s [mode]\n", os.Args[0])
		fmt.Fprintf(flag.CommandLine.Output(), "\n")
		fmt.Fprintf(flag.CommandLine.Output(), "Supported modes:")
		fmt.Fprintf(flag.CommandLine.Output(), "\n")
		fmt.Fprintf(flag.CommandLine.Output(), "* text2bytes - converts a text format trace to bytes\n")
		fmt.Fprintf(flag.CommandLine.Output(), "* bytes2text - converts a byte format trace to text\n")
		fmt.Fprintf(flag.CommandLine.Output(), "\n")
		flag.PrintDefaults()
	}
	log.SetFlags(0)
}

func main() {
	flag.Parse()
	if narg := flag.NArg(); narg != 1 {
		log.Fatal("expected exactly one positional argument: the mode to operate in; see -h output")
	}

	r := os.Stdin
	w := os.Stdout

	var tr traceReader
	var tw traceWriter
	var err error

	switch flag.Arg(0) {
	case "text2bytes":
		tr, err = raw.NewTextReader(r)
		if err != nil {
			log.Fatal(err)
		}
		tw, err = raw.NewWriter(w, tr.Version())
		if err != nil {
			log.Fatal(err)
		}
	case "bytes2text":
		tr, err = raw.NewReader(r)
		if err != nil {
			log.Fatal(err)
		}
		tw, err = raw.NewTextWriter(w, tr.Version())
		if err != nil {
			log.Fatal(err)
		}
	}
	for {
		ev, err := tr.ReadEvent()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
			break
		}
		if err := tw.WriteEvent(ev); err != nil {
			log.Fatal(err)
			break
		}
	}
}

type traceReader interface {
	Version() version.Version
	ReadEvent() (raw.Event, error)
}

type traceWriter interface {
	WriteEvent(raw.Event) error
}
