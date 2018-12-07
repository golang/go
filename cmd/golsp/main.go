// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The golsp command is an LSP server for Go.
// The Language Server Protocol allows any text editor
// to be extended with IDE-like features;
// see https://langserver.org/ for details.
package main // import "golang.org/x/tools/cmd/golsp"

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"time"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp"
)

var (
	cpuprofile = flag.String("cpuprofile", "", "write CPU profile to this file")
	memprofile = flag.String("memprofile", "", "write memory profile to this file")
	traceFlag  = flag.String("trace", "", "write trace log to this file")
	logfile    = flag.String("logfile", "", "filename to log to. if value is \"auto\", then logging to a default output file is enabled")

	// Flags for compatitibility with VSCode.
	mode = flag.String("mode", "", "no effect")
)

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: golsp [flags]\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() > 0 {
		flag.Usage()
		os.Exit(2)
	}

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal(err)
		}
		// NB: profile won't be written in case of error.
		defer pprof.StopCPUProfile()
	}

	if *traceFlag != "" {
		f, err := os.Create(*traceFlag)
		if err != nil {
			log.Fatal(err)
		}
		if err := trace.Start(f); err != nil {
			log.Fatal(err)
		}
		// NB: trace log won't be written in case of error.
		defer func() {
			trace.Stop()
			log.Printf("To view the trace, run:\n$ go tool trace view %s", *traceFlag)
		}()
	}

	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			log.Fatal(err)
		}
		// NB: memprofile won't be written in case of error.
		defer func() {
			runtime.GC() // get up-to-date statistics
			if err := pprof.WriteHeapProfile(f); err != nil {
				log.Fatalf("Writing memory profile: %v", err)
			}
			f.Close()
		}()
	}

	out := os.Stderr
	if *logfile != "" {
		filename := *logfile
		if filename == "auto" {
			filename = filepath.Join(os.TempDir(), fmt.Sprintf("golsp-%d.log", os.Getpid()))
		}
		f, err := os.Create(filename)
		if err != nil {
			log.Fatalf("Unable to create log file: %v", err)
		}
		defer f.Close()
		log.SetOutput(io.MultiWriter(os.Stderr, f))
		out = f
	}
	if err := lsp.RunServer(
		context.Background(),
		jsonrpc2.NewHeaderStream(os.Stdin, os.Stdout),
		func(direction jsonrpc2.Direction, id *jsonrpc2.ID, elapsed time.Duration, method string, payload *json.RawMessage, err *jsonrpc2.Error) {

			if err != nil {
				fmt.Fprintf(out, "[Error - %v] %s %s%s %v", time.Now().Format("3:04:05 PM"), direction, method, id, err)
				return
			}
			fmt.Fprintf(out, "[Trace - %v] ", time.Now().Format("3:04:05 PM"))
			switch direction {
			case jsonrpc2.Send:
				fmt.Fprint(out, "Received ")
			case jsonrpc2.Receive:
				fmt.Fprint(out, "Sending ")
			}
			switch {
			case id == nil:
				fmt.Fprint(out, "notification ")
			case elapsed >= 0:
				fmt.Fprint(out, "response ")
			default:
				fmt.Fprint(out, "request ")
			}
			fmt.Fprintf(out, "'%s", method)
			switch {
			case id == nil:
				// do nothing
			case id.Name != "":
				fmt.Fprintf(out, " - (%s)", id.Name)
			default:
				fmt.Fprintf(out, " - (%d)", id.Number)
			}
			fmt.Fprint(out, "'")
			if elapsed >= 0 {
				fmt.Fprintf(out, " in %vms", elapsed.Nanoseconds()/1000)
			}
			params := string(*payload)
			if params == "null" {
				params = "{}"
			}
			fmt.Fprintf(out, ".\r\nParams: %s\r\n\r\n\r\n", params)
		},
	); err != nil {
		log.Fatal(err)
	}
}
