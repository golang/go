// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// file2fuzz converts binary files, such as those used by go-fuzz, to the Go
// fuzzing corpus format.
//
// Usage:
//
//	file2fuzz [-o output] [input...]
//
// The defualt behavior is to read input from stdin and write the converted
// output to stdout. If any position arguments are provided stdin is ignored
// and the arguments are assumed to be input files to convert.
//
// The -o flag provides an path to write output files to. If only one positional
// argument is specified it may be a file path or an existing directory, if there are
// multiple inputs specified it must be a directory. If a directory is provided
// the name of the file will be the SHA-256 hash of its contents.
//
package main

import (
	"crypto/sha256"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
)

// encVersion1 is version 1 Go fuzzer corpus encoding.
var encVersion1 = "go test fuzz v1"

func encodeByteSlice(b []byte) []byte {
	return []byte(fmt.Sprintf("%s\n[]byte(%q)", encVersion1, b))
}

func usage() {
	fmt.Fprintf(os.Stderr, "usage: file2fuzz [-o output] [input...]\nconverts files to Go fuzzer corpus format\n")
	fmt.Fprintf(os.Stderr, "\tinput: files to convert\n")
	fmt.Fprintf(os.Stderr, "\t-o: where to write converted file(s)\n")
	os.Exit(2)
}
func dirWriter(dir string) func([]byte) error {
	return func(b []byte) error {
		sum := fmt.Sprintf("%x", sha256.Sum256(b))
		name := filepath.Join(dir, sum)
		if err := os.MkdirAll(dir, 0777); err != nil {
			return err
		}
		if err := ioutil.WriteFile(name, b, 0666); err != nil {
			os.Remove(name)
			return err
		}
		return nil
	}
}

func convert(inputArgs []string, outputArg string) error {
	var input []io.Reader
	if args := inputArgs; len(args) == 0 {
		input = []io.Reader{os.Stdin}
	} else {
		for _, a := range args {
			f, err := os.Open(a)
			if err != nil {
				return fmt.Errorf("unable to open %q: %s", a, err)
			}
			defer f.Close()
			if fi, err := f.Stat(); err != nil {
				return fmt.Errorf("unable to open %q: %s", a, err)
			} else if fi.IsDir() {
				return fmt.Errorf("%q is a directory, not a file", a)
			}
			input = append(input, f)
		}
	}

	var output func([]byte) error
	if outputArg == "" {
		if len(inputArgs) > 1 {
			return errors.New("-o required with multiple input files")
		}
		output = func(b []byte) error {
			_, err := os.Stdout.Write(b)
			return err
		}
	} else {
		if len(inputArgs) > 1 {
			output = dirWriter(outputArg)
		} else {
			if fi, err := os.Stat(outputArg); err != nil && !os.IsNotExist(err) {
				return fmt.Errorf("unable to open %q for writing: %s", outputArg, err)
			} else if err == nil && fi.IsDir() {
				output = dirWriter(outputArg)
			} else {
				output = func(b []byte) error {
					return ioutil.WriteFile(outputArg, b, 0666)
				}
			}
		}
	}

	for _, f := range input {
		b, err := ioutil.ReadAll(f)
		if err != nil {
			return fmt.Errorf("unable to read input: %s", err)
		}
		if err := output(encodeByteSlice(b)); err != nil {
			return fmt.Errorf("unable to write output: %s", err)
		}
	}

	return nil
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("file2fuzz: ")

	output := flag.String("o", "", "where to write converted file(s)")
	flag.Usage = usage
	flag.Parse()

	if err := convert(flag.Args(), *output); err != nil {
		log.Fatal(err)
	}
}
