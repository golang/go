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
	"internal/txtar"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
)

func main() {
	log.SetFlags(0)
	ctx, err := newContext()
	if err != nil {
		log.Fatal(err)
	}
	if err := ctx.runGenerators(); err != nil {
		log.Fatal(err)
	}
	if err := ctx.runTestProg("./testprog/annotations.go"); err != nil {
		log.Fatal(err)
	}
	if err := ctx.runTestProg("./testprog/annotations-stress.go"); err != nil {
		log.Fatal(err)
	}
}

type context struct {
	testNames map[string]struct{}
	filter    *regexp.Regexp
}

func newContext() (*context, error) {
	var filter *regexp.Regexp
	var err error
	if pattern := os.Getenv("GOTRACETEST"); pattern != "" {
		filter, err = regexp.Compile(pattern)
		if err != nil {
			return nil, fmt.Errorf("compiling regexp %q for GOTRACETEST: %v", pattern, err)
		}
	}
	return &context{
		testNames: make(map[string]struct{}),
		filter:    filter,
	}, nil
}

func (ctx *context) register(testName string) (skip bool, err error) {
	if _, ok := ctx.testNames[testName]; ok {
		return true, fmt.Errorf("duplicate test %s found", testName)
	}
	if ctx.filter != nil {
		return !ctx.filter.MatchString(testName), nil
	}
	return false, nil
}

func (ctx *context) runGenerators() error {
	generators, err := filepath.Glob("./generators/*.go")
	if err != nil {
		return fmt.Errorf("reading generators: %v", err)
	}
	genroot := "./tests"

	if err := os.MkdirAll(genroot, 0777); err != nil {
		return fmt.Errorf("creating generated root: %v", err)
	}
	for _, path := range generators {
		name := filepath.Base(path)
		name = name[:len(name)-len(filepath.Ext(name))]

		// Skip if we have a pattern and this test doesn't match.
		skip, err := ctx.register(name)
		if err != nil {
			return err
		}
		if skip {
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

func (ctx *context) runTestProg(progPath string) error {
	name := filepath.Base(progPath)
	name = name[:len(name)-len(filepath.Ext(name))]
	name = fmt.Sprintf("go1%d-%s", version.Current, name)

	// Skip if we have a pattern and this test doesn't match.
	skip, err := ctx.register(name)
	if err != nil {
		return err
	}
	if skip {
		return nil
	}

	// Create command.
	var trace, stderr bytes.Buffer
	cmd := exec.Command("go", "run", progPath)
	// TODO(mknyszek): Remove if goexperiment.Exectracer2 becomes the default.
	cmd.Env = append(os.Environ(), "GOEXPERIMENT=exectracer2")
	cmd.Stdout = &trace
	cmd.Stderr = &stderr

	// Run trace program; the trace will appear in stdout.
	fmt.Fprintf(os.Stderr, "running trace program %s...\n", name)
	if err := cmd.Run(); err != nil {
		log.Fatalf("running trace program: %v:\n%s", err, stderr.String())
	}

	// Write out the trace.
	var textTrace bytes.Buffer
	r, err := raw.NewReader(&trace)
	if err != nil {
		log.Fatalf("reading trace: %v", err)
	}
	w, err := raw.NewTextWriter(&textTrace, version.Current)
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
	testData := txtar.Format(&txtar.Archive{
		Files: []txtar.File{
			{Name: "expect", Data: []byte("SUCCESS")},
			{Name: "trace", Data: textTrace.Bytes()},
		},
	})
	return os.WriteFile(fmt.Sprintf("./tests/%s.test", name), testData, 0o664)
}
