// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// Interactive debugging shell for codehost.Repo implementations.

package main

import (
	"archive/zip"
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strings"
	"time"

	"cmd/go/internal/modfetch/codehost"
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: go run shell.go vcs remote\n")
	os.Exit(2)
}

func main() {
	codehost.WorkRoot = "/tmp/vcswork"
	log.SetFlags(0)
	log.SetPrefix("shell: ")
	flag.Usage = usage
	flag.Parse()
	if flag.NArg() != 2 {
		usage()
	}

	repo, err := codehost.NewRepo(flag.Arg(0), flag.Arg(1))
	if err != nil {
		log.Fatal(err)
	}

	b := bufio.NewReader(os.Stdin)
	for {
		fmt.Fprintf(os.Stderr, ">>> ")
		line, err := b.ReadString('\n')
		if err != nil {
			log.Fatal(err)
		}
		f := strings.Fields(line)
		if len(f) == 0 {
			continue
		}
		switch f[0] {
		default:
			fmt.Fprintf(os.Stderr, "?unknown command\n")
			continue
		case "tags":
			prefix := ""
			if len(f) == 2 {
				prefix = f[1]
			}
			if len(f) > 2 {
				fmt.Fprintf(os.Stderr, "?usage: tags [prefix]\n")
				continue
			}
			tags, err := repo.Tags(prefix)
			if err != nil {
				fmt.Fprintf(os.Stderr, "?%s\n", err)
				continue
			}
			for _, tag := range tags {
				fmt.Printf("%s\n", tag)
			}

		case "stat":
			if len(f) != 2 {
				fmt.Fprintf(os.Stderr, "?usage: stat rev\n")
				continue
			}
			info, err := repo.Stat(f[1])
			if err != nil {
				fmt.Fprintf(os.Stderr, "?%s\n", err)
				continue
			}
			fmt.Printf("name=%s short=%s version=%s time=%s\n", info.Name, info.Short, info.Version, info.Time.UTC().Format(time.RFC3339))

		case "read":
			if len(f) != 3 {
				fmt.Fprintf(os.Stderr, "?usage: read rev file\n")
				continue
			}
			data, err := repo.ReadFile(f[1], f[2], 10<<20)
			if err != nil {
				fmt.Fprintf(os.Stderr, "?%s\n", err)
				continue
			}
			os.Stdout.Write(data)

		case "zip":
			if len(f) != 4 {
				fmt.Fprintf(os.Stderr, "?usage: zip rev subdir output\n")
				continue
			}
			subdir := f[2]
			if subdir == "-" {
				subdir = ""
			}
			rc, _, err := repo.ReadZip(f[1], subdir, 10<<20)
			if err != nil {
				fmt.Fprintf(os.Stderr, "?%s\n", err)
				continue
			}
			data, err := ioutil.ReadAll(rc)
			rc.Close()
			if err != nil {
				fmt.Fprintf(os.Stderr, "?%s\n", err)
				continue
			}

			if f[3] != "-" {
				if err := ioutil.WriteFile(f[3], data, 0666); err != nil {
					fmt.Fprintf(os.Stderr, "?%s\n", err)
					continue
				}
			}
			z, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
			if err != nil {
				fmt.Fprintf(os.Stderr, "?%s\n", err)
				continue
			}
			for _, f := range z.File {
				fmt.Printf("%s %d\n", f.Name, f.UncompressedSize64)
			}
		}
	}
}
