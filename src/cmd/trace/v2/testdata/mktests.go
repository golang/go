// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

package main

import (
	"bytes"
	"fmt"
	"internal/trace/v2/raw"
	"internal/trace/v2/version"
	"io"
	"log"
	"os"
	"os/exec"
)

func main() {
	// Create command.
	var trace, stderr bytes.Buffer
	cmd := exec.Command("go", "run", "./testprog/main.go")
	cmd.Stdout = &trace
	cmd.Stderr = &stderr

	// Run trace program; the trace will appear in stdout.
	fmt.Fprintln(os.Stderr, "running trace program...")
	if err := cmd.Run(); err != nil {
		log.Fatalf("running trace program: %v:\n%s", err, stderr.String())
	}

	// Create file.
	f, err := os.Create(fmt.Sprintf("./go1%d.test", version.Current))
	if err != nil {
		log.Fatalf("creating output file: %v", err)
	}
	defer f.Close()

	// Write out the trace.
	r, err := raw.NewReader(&trace)
	if err != nil {
		log.Fatalf("reading trace: %v", err)
	}
	w, err := raw.NewTextWriter(f, version.Current)
	for {
		ev, err := r.ReadEvent()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatalf("reading trace: %v", err)
		}
		if err := w.WriteEvent(ev); err != nil {
			log.Fatalf("writing trace: %v", err)
		}
	}
}
