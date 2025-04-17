// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"internal/coverage/slicewriter"
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime/coverage"
	"strings"
)

var verbflag = flag.Int("v", 0, "Verbose trace output level")
var testpointflag = flag.String("tp", "", "Testpoint to run")
var outdirflag = flag.String("o", "", "Output dir into which to emit")

func emitToWriter() {
	log.SetPrefix("emitToWriter: ")
	var slwm slicewriter.WriteSeeker
	if err := coverage.WriteMeta(&slwm); err != nil {
		log.Fatalf("error: WriteMeta returns %v", err)
	}
	mf := filepath.Join(*outdirflag, "covmeta.0abcdef")
	if err := os.WriteFile(mf, slwm.BytesWritten(), 0666); err != nil {
		log.Fatalf("error: writing %s: %v", mf, err)
	}
	var slwc slicewriter.WriteSeeker
	if err := coverage.WriteCounters(&slwc); err != nil {
		log.Fatalf("error: WriteCounters returns %v", err)
	}
	cf := filepath.Join(*outdirflag, "covcounters.0abcdef.99.77")
	if err := os.WriteFile(cf, slwc.BytesWritten(), 0666); err != nil {
		log.Fatalf("error: writing %s: %v", cf, err)
	}
}

func emitToDir() {
	log.SetPrefix("emitToDir: ")
	if err := coverage.WriteMetaDir(*outdirflag); err != nil {
		log.Fatalf("error: WriteMetaDir returns %v", err)
	}
	if err := coverage.WriteCountersDir(*outdirflag); err != nil {
		log.Fatalf("error: WriteCountersDir returns %v", err)
	}
}

func emitToNonexistentDir() {
	log.SetPrefix("emitToNonexistentDir: ")

	want := []string{
		"no such file or directory",             // linux-ish
		"system cannot find the file specified", // windows
		"does not exist",                        // plan9
	}

	checkWant := func(which string, got string) {
		found := false
		for _, w := range want {
			if strings.Contains(got, w) {
				found = true
				break
			}
		}
		if !found {
			log.Fatalf("%s emit to bad dir: got error:\n  %v\nwanted error with one of:\n  %+v", which, got, want)
		}
	}

	// Mangle the output directory to produce something nonexistent.
	mangled := *outdirflag + "_MANGLED"
	if err := coverage.WriteMetaDir(mangled); err == nil {
		log.Fatal("expected error from WriteMetaDir to nonexistent dir")
	} else {
		got := fmt.Sprintf("%v", err)
		checkWant("meta data", got)
	}

	// Now try to emit counter data file to a bad dir.
	if err := coverage.WriteCountersDir(mangled); err == nil {
		log.Fatal("expected error emitting counter data to bad dir")
	} else {
		got := fmt.Sprintf("%v", err)
		checkWant("counter data", got)
	}
}

func emitToUnwritableDir() {
	log.SetPrefix("emitToUnwritableDir: ")

	want := "permission denied"

	if err := coverage.WriteMetaDir(*outdirflag); err == nil {
		log.Fatal("expected error from WriteMetaDir to unwritable dir")
	} else {
		got := fmt.Sprintf("%v", err)
		if !strings.Contains(got, want) {
			log.Fatalf("meta-data emit to unwritable dir: wanted error containing %q got %q", want, got)
		}
	}

	// Similarly with writing counter data.
	if err := coverage.WriteCountersDir(*outdirflag); err == nil {
		log.Fatal("expected error emitting counter data to unwritable dir")
	} else {
		got := fmt.Sprintf("%v", err)
		if !strings.Contains(got, want) {
			log.Fatalf("emitting counter data to unwritable dir: wanted error containing %q got %q", want, got)
		}
	}
}

func emitToNilWriter() {
	log.SetPrefix("emitToWriter: ")
	want := "nil writer"
	var bad io.WriteSeeker
	if err := coverage.WriteMeta(bad); err == nil {
		log.Fatal("expected error passing nil writer for meta emit")
	} else {
		got := fmt.Sprintf("%v", err)
		if !strings.Contains(got, want) {
			log.Fatalf("emitting meta-data passing nil writer: wanted error containing %q got %q", want, got)
		}
	}

	if err := coverage.WriteCounters(bad); err == nil {
		log.Fatal("expected error passing nil writer for counter emit")
	} else {
		got := fmt.Sprintf("%v", err)
		if !strings.Contains(got, want) {
			log.Fatalf("emitting counter data passing nil writer: wanted error containing %q got %q", want, got)
		}
	}
}

type failingWriter struct {
	writeCount int
	writeLimit int
	slws       slicewriter.WriteSeeker
}

func (f *failingWriter) Write(p []byte) (n int, err error) {
	c := f.writeCount
	f.writeCount++
	if f.writeLimit < 0 || c < f.writeLimit {
		return f.slws.Write(p)
	}
	return 0, fmt.Errorf("manufactured write error")
}

func (f *failingWriter) Seek(offset int64, whence int) (int64, error) {
	return f.slws.Seek(offset, whence)
}

func (f *failingWriter) reset(lim int) {
	f.writeCount = 0
	f.writeLimit = lim
	f.slws = slicewriter.WriteSeeker{}
}

func writeStressTest(tag string, testf func(testf *failingWriter) error) {
	// Invoke the function initially without the write limit
	// set, to capture the number of writes performed.
	fw := &failingWriter{writeLimit: -1}
	testf(fw)

	// Now that we know how many writes are going to happen, run the
	// function repeatedly, each time with a Write operation set to
	// fail at a new spot. The goal here is to make sure that:
	// A) an error is reported, and B) nothing crashes.
	tot := fw.writeCount
	for i := 0; i < tot; i++ {
		fw.reset(i)
		err := testf(fw)
		if err == nil {
			log.Fatalf("no error from write %d tag %s", i, tag)
		}
	}
}

func postClear() int {
	return 42
}

func preClear() int {
	return 42
}

// This test is designed to ensure that write errors are properly
// handled by the code that writes out coverage data. It repeatedly
// invokes the 'emit to writer' apis using a specially crafted writer
// that captures the total number of expected writes, then replays the
// execution N times with a manufactured write error at the
// appropriate spot.
func emitToFailingWriter() {
	log.SetPrefix("emitToFailingWriter: ")

	writeStressTest("emit-meta", func(f *failingWriter) error {
		return coverage.WriteMeta(f)
	})
	writeStressTest("emit-counter", func(f *failingWriter) error {
		return coverage.WriteCounters(f)
	})
}

func emitWithCounterClear() {
	log.SetPrefix("emitWitCounterClear: ")
	preClear()
	if err := coverage.ClearCounters(); err != nil {
		log.Fatalf("clear failed: %v", err)
	}
	postClear()
	if err := coverage.WriteMetaDir(*outdirflag); err != nil {
		log.Fatalf("error: WriteMetaDir returns %v", err)
	}
	if err := coverage.WriteCountersDir(*outdirflag); err != nil {
		log.Fatalf("error: WriteCountersDir returns %v", err)
	}
}

func final() int {
	println("I run last.")
	return 43
}

func main() {
	log.SetFlags(0)
	flag.Parse()
	if *testpointflag == "" {
		log.Fatalf("error: no testpoint (use -tp flag)")
	}
	if *outdirflag == "" {
		log.Fatalf("error: no output dir specified (use -o flag)")
	}
	switch *testpointflag {
	case "emitToDir":
		emitToDir()
	case "emitToWriter":
		emitToWriter()
	case "emitToNonexistentDir":
		emitToNonexistentDir()
	case "emitToUnwritableDir":
		emitToUnwritableDir()
	case "emitToNilWriter":
		emitToNilWriter()
	case "emitToFailingWriter":
		emitToFailingWriter()
	case "emitWithCounterClear":
		emitWithCounterClear()
	default:
		log.Fatalf("error: unknown testpoint %q", *testpointflag)
	}
	final()
}
