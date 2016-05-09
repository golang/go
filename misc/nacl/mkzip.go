// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Mkzip creates a zip file from a 'proto' file describing the contents.
//
// The proto file is inspired by the Plan 9 mkfs prototype file format.
// It describes a file tree, one directory per line, with leading tab
// indentation marking the tree structure. Each line contains a leading
// name field giving the name of the file to copy into the zip file,
// and then a sequence of optional key=value attributes to control
// the copy. The only known attribute is src=foo, meaning copy the
// actual data for the file (or directory) from an alternate location.
package main

import (
	"archive/zip"
	"bufio"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path"
	"path/filepath"
	"strings"
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: mkzip [-r root] src.proto out.zip\n")
	os.Exit(2)
}

func sysfatal(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "mkzip: %s\n", fmt.Sprintf(format, args...))
	os.Exit(2)
}

var (
	root      = flag.String("r", ".", "interpret source paths relative to this directory")
	gopackage = flag.String("p", "", "write Go source file in this package")
)

type stack struct {
	name  string
	src   string
	depth int
}

func main() {
	log.SetFlags(0)
	flag.Usage = usage
	flag.Parse()

	args := flag.Args()
	if len(args) != 2 {
		usage()
	}

	rf, err := os.Open(args[0])
	if err != nil {
		sysfatal("%v", err)
	}
	r := bufio.NewScanner(rf)

	zf, err := os.Create(args[1])
	if err != nil {
		sysfatal("%v", err)
	}

	var w io.Writer = zf
	if *gopackage != "" {
		fmt.Fprintf(zf, `package %s
import "sync"
func init() {
	var once sync.Once
	fsinit = func() {
		once.Do(func() {
			unzip("`, *gopackage)
		gw := &goWriter{b: bufio.NewWriter(w)}
		defer func() {
			if err := gw.Close(); err != nil {
				sysfatal("finishing Go output: %v", err)
			}
		}()
		w = gw
	}
	z := zip.NewWriter(w)

	lineno := 0

	addfile := func(info os.FileInfo, dst string, src string) {
		zh, err := zip.FileInfoHeader(info)
		if err != nil {
			sysfatal("%s:%d: %s: %v", args[0], lineno, src, err)
		}
		zh.Name = dst
		zh.Method = zip.Deflate
		if info.IsDir() && !strings.HasSuffix(dst, "/") {
			zh.Name += "/"
		}
		w, err := z.CreateHeader(zh)
		if err != nil {
			sysfatal("%s:%d: %s: %v", args[0], lineno, src, err)
		}
		if info.IsDir() {
			return
		}
		r, err := os.Open(src)
		if err != nil {
			sysfatal("%s:%d: %s: %v", args[0], lineno, src, err)
		}
		defer r.Close()
		if _, err := io.Copy(w, r); err != nil {
			sysfatal("%s:%d: %s: %v", args[0], lineno, src, err)
		}
	}

	var stk []stack

	for r.Scan() {
		line := r.Text()
		lineno++
		s := strings.TrimLeft(line, "\t")
		prefix, line := line[:len(line)-len(s)], s
		if i := strings.Index(line, "#"); i >= 0 {
			line = line[:i]
		}
		f := strings.Fields(line)
		if len(f) == 0 {
			continue
		}
		if strings.HasPrefix(line, " ") {
			sysfatal("%s:%d: must use tabs for indentation", args[0], lineno)
		}
		depth := len(prefix)
		for len(stk) > 0 && depth <= stk[len(stk)-1].depth {
			stk = stk[:len(stk)-1]
		}
		parent := ""
		psrc := *root
		if len(stk) > 0 {
			parent = stk[len(stk)-1].name
			psrc = stk[len(stk)-1].src
		}
		if strings.Contains(f[0], "/") {
			sysfatal("%s:%d: destination name cannot contain slash", args[0], lineno)
		}
		name := path.Join(parent, f[0])
		src := filepath.Join(psrc, f[0])
		for _, attr := range f[1:] {
			i := strings.Index(attr, "=")
			if i < 0 {
				sysfatal("%s:%d: malformed attribute %q", args[0], lineno, attr)
			}
			key, val := attr[:i], attr[i+1:]
			switch key {
			case "src":
				src = val
			default:
				sysfatal("%s:%d: unknown attribute %q", args[0], lineno, attr)
			}
		}

		stk = append(stk, stack{name: name, src: src, depth: depth})

		if f[0] == "*" || f[0] == "+" {
			if f[0] == "*" {
				dir, err := ioutil.ReadDir(psrc)
				if err != nil {
					sysfatal("%s:%d: %v", args[0], lineno, err)
				}
				for _, d := range dir {
					addfile(d, path.Join(parent, d.Name()), filepath.Join(psrc, d.Name()))
				}
			} else {
				err := filepath.Walk(psrc, func(src string, info os.FileInfo, err error) error {
					if err != nil {
						return err
					}
					if src == psrc {
						return nil
					}
					if psrc == "." {
						psrc = ""
					}
					name := path.Join(parent, filepath.ToSlash(src[len(psrc):]))
					addfile(info, name, src)
					return nil
				})
				if err != nil {
					sysfatal("%s:%d: %v", args[0], lineno, err)
				}
			}
			continue
		}

		fi, err := os.Stat(src)
		if err != nil {
			sysfatal("%s:%d: %v", args[0], lineno, err)
		}
		addfile(fi, name, src)
	}

	if err := z.Close(); err != nil {
		sysfatal("finishing zip file: %v", err)
	}
}

type goWriter struct {
	b *bufio.Writer
}

func (w *goWriter) Write(b []byte) (int, error) {
	for _, c := range b {
		fmt.Fprintf(w.b, "\\x%02x", c)
	}
	return len(b), nil
}

func (w *goWriter) Close() error {
	fmt.Fprintf(w.b, "\")\n\t\t})\n\t}\n}")
	w.b.Flush()
	return nil
}
