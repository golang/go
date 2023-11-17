// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
)

func main() {
	log.SetFlags(0)
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	generators, err := filepath.Glob("./generators/*.go")
	if err != nil {
		return fmt.Errorf("reading generators: %v", err)
	}
	genroot := "./tests"

	// Grab a pattern, if any.
	var re *regexp.Regexp
	if pattern := os.Getenv("GOTRACETEST"); pattern != "" {
		re, err = regexp.Compile(pattern)
		if err != nil {
			return fmt.Errorf("compiling regexp %q for GOTRACETEST: %v", pattern, err)
		}
	}

	if err := os.MkdirAll(genroot, 0777); err != nil {
		return fmt.Errorf("creating generated root: %v", err)
	}
	for _, path := range generators {
		name := filepath.Base(path)
		name = name[:len(name)-len(filepath.Ext(name))]

		// Skip if we have a pattern and this test doesn't match.
		if re != nil && !re.MatchString(name) {
			continue
		}

		fmt.Fprintf(os.Stderr, "generating %s... ", name)

		// Get the test path.
		testPath := filepath.Join(genroot, fmt.Sprintf("%s.test", name))

		// Run generator.
		cmd := exec.Command("go", "run", path, testPath)
		if out, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("running generator %s: %v:\n%s", name, err, out)
		}
		fmt.Fprintln(os.Stderr)
	}
	return nil
}
