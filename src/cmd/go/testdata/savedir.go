// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// Savedir archives a directory tree as a txtar archive printed to standard output.
//
// Usage:
//
//	go run savedir.go /path/to/dir >saved.txt
//
// Typically the tree is later extracted during a test with tg.extract("testdata/saved.txt").
//
package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"../internal/txtar"
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: go run savedir.go dir >saved.txt\n")
	os.Exit(2)
}

const goCmd = "vgo"

func main() {
	flag.Usage = usage
	flag.Parse()
	if flag.NArg() != 1 {
		usage()
	}

	log.SetPrefix("savedir: ")
	log.SetFlags(0)

	dir := flag.Arg(0)

	a := new(txtar.Archive)
	dir = filepath.Clean(dir)
	filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if path == dir {
			return nil
		}
		name := info.Name()
		if strings.HasPrefix(name, ".") {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if !info.Mode().IsRegular() {
			return nil
		}
		data, err := ioutil.ReadFile(path)
		if err != nil {
			log.Fatal(err)
		}
		if !utf8.Valid(data) {
			log.Printf("%s: ignoring invalid UTF-8 data", path)
			return nil
		}
		a.Files = append(a.Files, txtar.File{Name: strings.TrimPrefix(path, dir+string(filepath.Separator)), Data: data})
		return nil
	})

	data := txtar.Format(a)
	os.Stdout.Write(data)
}
