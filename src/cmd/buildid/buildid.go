// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"cmd/internal/buildid"
	"cmd/internal/telemetry"
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: go tool buildid [-w] file\n")
	flag.PrintDefaults()
	os.Exit(2)
}

var wflag = flag.Bool("w", false, "write build ID")

func main() {
	log.SetPrefix("buildid: ")
	log.SetFlags(0)
	telemetry.Start()
	flag.Usage = usage
	flag.Parse()
	telemetry.Inc("buildid/invocations")
	telemetry.CountFlags("buildid/flag:", *flag.CommandLine)
	if flag.NArg() != 1 {
		usage()
	}

	file := flag.Arg(0)
	id, err := buildid.ReadFile(file)
	if err != nil {
		log.Fatal(err)
	}
	if !*wflag {
		fmt.Printf("%s\n", id)
		return
	}

	// Keep in sync with src/cmd/go/internal/work/buildid.go:updateBuildID

	f, err := os.Open(file)
	if err != nil {
		log.Fatal(err)
	}
	matches, hash, err := buildid.FindAndHash(f, id, 0)
	f.Close()
	if err != nil {
		log.Fatal(err)
	}

	// <= go 1.7 doesn't embed the contentID or actionID, so no slash is present
	if !strings.Contains(id, "/") {
		log.Fatalf("%s: build ID is a legacy format...binary too old for this tool", file)
	}

	newID := id[:strings.LastIndex(id, "/")] + "/" + buildid.HashToString(hash)
	if len(newID) != len(id) {
		log.Fatalf("%s: build ID length mismatch %q vs %q", file, id, newID)
	}

	if len(matches) == 0 {
		return
	}

	f, err = os.OpenFile(file, os.O_RDWR, 0)
	if err != nil {
		log.Fatal(err)
	}
	if err := buildid.Rewrite(f, matches, newID); err != nil {
		log.Fatal(err)
	}
	if err := f.Close(); err != nil {
		log.Fatal(err)
	}
}
