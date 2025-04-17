// buildrun -t 45

//go:build !js && !wasip1

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Ensure that runtime traceback does not infinite loop for
// the testcase below.

package main

import (
	"bytes"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
)

const prog = `

package main

import "context"

var gpi *int

type nAO struct {
	eE bool
}

type NAO func(*nAO)

func WEA() NAO {
	return func(o *nAO) { o.eE = true }
}

type R struct {
	cM *CM
}

type CM int

type A string

func (m *CM) NewA(ctx context.Context, cN string, nn *nAO, opts ...NAO) (*A, error) {
	for _, o := range opts {
		o(nn)
	}
	s := A("foo")
	return &s, nil
}

func (r *R) CA(ctx context.Context, cN string, nn *nAO) (*int, error) {
	cA, err := r.cM.NewA(ctx, cN, nn, WEA(), WEA())
	if err == nil {
		return nil, err
	}
	println(cA)
	x := int(42)
	return &x, nil
}

func main() {
	c := CM(1)
	r := R{cM: &c}
	var ctx context.Context
	nnr := nAO{}
	pi, err := r.CA(ctx, "foo", nil)
	if err != nil {
		panic("bad")
	}
	println(nnr.eE)
	gpi = pi
}
`

func main() {
	dir, err := ioutil.TempDir("", "46234")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	file := filepath.Join(dir, "main.go")
	if err := ioutil.WriteFile(file, []byte(prog), 0655); err != nil {
		log.Fatalf("Write error %v", err)
	}

	cmd := exec.Command("go", "run", file)
	output, err := cmd.CombinedOutput()
	if err == nil {
		log.Fatalf("Passed, expected an error")
	}

	want := []byte("nil pointer dereference")
	if !bytes.Contains(output, want) {
		log.Fatalf("Unmatched error message %q:\nin\n%s\nError: %v", want, output, err)
	}
}
