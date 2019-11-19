// +build !nacl,!js
// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Ensure that deadlock detection can still
// run even with an import of "_ os/signal".

package main

import (
	"bytes"
	"context"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

const prog = `
package main

import _ "os/signal"

func main() {
  c := make(chan int)
  c <- 1
}
`

func main() {
	dir, err := ioutil.TempDir("", "21576")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	file := filepath.Join(dir, "main.go")
	if err := ioutil.WriteFile(file, []byte(prog), 0655); err != nil {
		log.Fatalf("Write error %v", err)
	}

	// Using a timeout of 1 minute in case other factors might slow
	// down the start of "go run". See https://golang.org/issue/34836.
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()

	cmd := exec.CommandContext(ctx, "go", "run", file)
	output, err := cmd.CombinedOutput()
	if err == nil {
		log.Fatalf("Passed, expected an error")
	}

	want := []byte("fatal error: all goroutines are asleep - deadlock!")
	if !bytes.Contains(output, want) {
		log.Fatalf("Unmatched error message %q:\nin\n%s\nError: %v", want, output, err)
	}
}
