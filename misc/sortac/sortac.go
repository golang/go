// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Sortac sorts the AUTHORS and CONTRIBUTORS files.
//
// Usage:
//
//    sortac [file...]
//
// Sortac sorts the named files in place.
// If given no arguments, it sorts standard input to standard output.
package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"

	"golang.org/x/text/collate"
	"golang.org/x/text/language"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("sortac: ")
	flag.Parse()

	args := flag.Args()
	if len(args) == 0 {
		os.Stdout.Write(sortAC(os.Stdin))
	} else {
		for _, arg := range args {
			f, err := os.Open(arg)
			if err != nil {
				log.Fatal(err)
			}
			sorted := sortAC(f)
			f.Close()
			if err := ioutil.WriteFile(arg, sorted, 0644); err != nil {
				log.Fatal(err)
			}
		}
	}
}

func sortAC(r io.Reader) []byte {
	bs := bufio.NewScanner(r)
	var header []string
	var lines []string
	for bs.Scan() {
		t := bs.Text()
		lines = append(lines, t)
		if t == "# Please keep the list sorted." {
			header = lines
			lines = nil
			continue
		}
	}
	if err := bs.Err(); err != nil {
		log.Fatal(err)
	}

	var out bytes.Buffer
	c := collate.New(language.Und, collate.Loose)
	c.SortStrings(lines)
	for _, l := range header {
		fmt.Fprintln(&out, l)
	}
	for _, l := range lines {
		fmt.Fprintln(&out, l)
	}
	return out.Bytes()
}
