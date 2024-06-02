// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmp"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"slices"
	"text/tabwriter"

	"internal/trace/event"
	"internal/trace/raw"
)

func init() {
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage: %s [mode]\n", os.Args[0])
		fmt.Fprintf(flag.CommandLine.Output(), "\n")
		fmt.Fprintf(flag.CommandLine.Output(), "Accepts a trace at stdin.\n")
		fmt.Fprintf(flag.CommandLine.Output(), "\n")
		fmt.Fprintf(flag.CommandLine.Output(), "Supported modes:")
		fmt.Fprintf(flag.CommandLine.Output(), "\n")
		fmt.Fprintf(flag.CommandLine.Output(), "* size  - dumps size stats\n")
		fmt.Fprintf(flag.CommandLine.Output(), "\n")
		flag.PrintDefaults()
	}
	log.SetFlags(0)
}

func main() {
	log.SetPrefix("")
	flag.Parse()

	if flag.NArg() != 1 {
		log.Print("missing mode argument")
		flag.Usage()
		os.Exit(1)
	}
	var err error
	switch mode := flag.Arg(0); mode {
	case "size":
		err = printSizeStats(os.Stdin)
	default:
		log.Printf("unknown mode %q", mode)
		flag.Usage()
		os.Exit(1)
	}
	if err != nil {
		log.Fatalf("error: %v", err)
		os.Exit(1)
	}
}

func printSizeStats(r io.Reader) error {
	cr := countingReader{Reader: r}
	tr, err := raw.NewReader(&cr)
	if err != nil {
		return err
	}
	type eventStats struct {
		typ   event.Type
		count int
		bytes int
	}
	var stats [256]eventStats
	for i := range stats {
		stats[i].typ = event.Type(i)
	}
	eventsRead := 0
	for {
		e, err := tr.ReadEvent()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		s := &stats[e.Ev]
		s.count++
		s.bytes += encodedSize(&e)
		eventsRead++
	}
	slices.SortFunc(stats[:], func(a, b eventStats) int {
		return cmp.Compare(b.bytes, a.bytes)
	})
	specs := tr.Version().Specs()
	w := tabwriter.NewWriter(os.Stdout, 3, 8, 2, ' ', 0)
	fmt.Fprintf(w, "Event\tBytes\t%%\tCount\t%%\n")
	fmt.Fprintf(w, "-\t-\t-\t-\t-\n")
	for i := range stats {
		stat := &stats[i]
		name := ""
		if int(stat.typ) >= len(specs) {
			name = fmt.Sprintf("<unknown (%d)>", stat.typ)
		} else {
			name = specs[stat.typ].Name
		}
		bytesPct := float64(stat.bytes) / float64(cr.bytesRead) * 100
		countPct := float64(stat.count) / float64(eventsRead) * 100
		fmt.Fprintf(w, "%s\t%d\t%.2f%%\t%d\t%.2f%%\n", name, stat.bytes, bytesPct, stat.count, countPct)
	}
	w.Flush()
	return nil
}

func encodedSize(e *raw.Event) int {
	size := 1
	var buf [binary.MaxVarintLen64]byte
	for _, arg := range e.Args {
		size += binary.PutUvarint(buf[:], arg)
	}
	spec := e.Version.Specs()[e.Ev]
	if spec.HasData {
		size += binary.PutUvarint(buf[:], uint64(len(e.Data)))
		size += len(e.Data)
	}
	return size
}

type countingReader struct {
	io.Reader
	bytesRead int
}

func (r *countingReader) Read(b []byte) (int, error) {
	n, err := r.Reader.Read(b)
	r.bytesRead += n
	return n, err
}
