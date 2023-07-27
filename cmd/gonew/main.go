// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Gonew starts a new Go module by copying a template module.
//
// Usage:
//
//	gonew srcmod[@version] [dstmod [dir]]
//
// Gonew makes a copy of the srcmod module, changing its module path to dstmod.
// It writes that new module to a new directory named by dir.
// If dir already exists, it must be an empty directory.
// If dir is omitted, gonew uses ./elem where elem is the final path element of dstmod.
//
// This command is highly experimental and subject to change.
//
// # Example
//
// To install gonew:
//
//	go install golang.org/x/tools/cmd/gonew@latest
//
// To clone the basic command-line program template golang.org/x/example/hello
// as your.domain/myprog, in the directory ./myprog:
//
//	gonew golang.org/x/example/hello your.domain/myprog
//
// To clone the latest copy of the rsc.io/quote module, keeping that module path,
// into ./quote:
//
//	gonew rsc.io/quote
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/parser"
	"go/token"
	"io/fs"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
	"golang.org/x/tools/internal/edit"
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: gonew srcmod[@version] [dstmod [dir]]\n")
	fmt.Fprintf(os.Stderr, "See https://pkg.go.dev/golang.org/x/tools/cmd/gonew.\n")
	os.Exit(2)
}

func main() {
	log.SetPrefix("gonew: ")
	log.SetFlags(0)
	flag.Usage = usage
	flag.Parse()
	args := flag.Args()

	if len(args) < 1 || len(args) > 3 {
		usage()
	}

	srcMod := args[0]
	srcModVers := srcMod
	if !strings.Contains(srcModVers, "@") {
		srcModVers += "@latest"
	}
	srcMod, _, _ = strings.Cut(srcMod, "@")
	if err := module.CheckPath(srcMod); err != nil {
		log.Fatalf("invalid source module name: %v", err)
	}

	dstMod := srcMod
	if len(args) >= 2 {
		dstMod = args[1]
		if err := module.CheckPath(dstMod); err != nil {
			log.Fatalf("invalid destination module name: %v", err)
		}
	}

	var dir string
	if len(args) == 3 {
		dir = args[2]
	} else {
		dir = "." + string(filepath.Separator) + path.Base(dstMod)
	}

	// Dir must not exist or must be an empty directory.
	de, err := os.ReadDir(dir)
	if err == nil && len(de) > 0 {
		log.Fatalf("target directory %s exists and is non-empty", dir)
	}
	needMkdir := err != nil

	var stdout, stderr bytes.Buffer
	cmd := exec.Command("go", "mod", "download", "-json", srcModVers)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		log.Fatalf("go mod download -json %s: %v\n%s%s", srcModVers, err, stderr.Bytes(), stdout.Bytes())
	}

	var info struct {
		Dir string
	}
	if err := json.Unmarshal(stdout.Bytes(), &info); err != nil {
		log.Fatalf("go mod download -json %s: invalid JSON output: %v\n%s%s", srcMod, err, stderr.Bytes(), stdout.Bytes())
	}

	if needMkdir {
		if err := os.MkdirAll(dir, 0777); err != nil {
			log.Fatal(err)
		}
	}

	// Copy from module cache into new directory, making edits as needed.
	filepath.WalkDir(info.Dir, func(src string, d fs.DirEntry, err error) error {
		if err != nil {
			log.Fatal(err)
		}
		rel, err := filepath.Rel(info.Dir, src)
		if err != nil {
			log.Fatal(err)
		}
		dst := filepath.Join(dir, rel)
		if d.IsDir() {
			if err := os.MkdirAll(dst, 0777); err != nil {
				log.Fatal(err)
			}
			return nil
		}

		data, err := os.ReadFile(src)
		if err != nil {
			log.Fatal(err)
		}

		isRoot := !strings.Contains(rel, string(filepath.Separator))
		if strings.HasSuffix(rel, ".go") {
			data = fixGo(data, rel, srcMod, dstMod, isRoot)
		}
		if rel == "go.mod" {
			data = fixGoMod(data, srcMod, dstMod)
		}

		if err := os.WriteFile(dst, data, 0666); err != nil {
			log.Fatal(err)
		}
		return nil
	})

	log.Printf("initialized %s in %s", dstMod, dir)
}

// fixGo rewrites the Go source in data to replace srcMod with dstMod.
// isRoot indicates whether the file is in the root directory of the module,
// in which case we also update the package name.
func fixGo(data []byte, file string, srcMod, dstMod string, isRoot bool) []byte {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, file, data, parser.ImportsOnly)
	if err != nil {
		log.Fatalf("parsing source module:\n%s", err)
	}

	buf := edit.NewBuffer(data)
	at := func(p token.Pos) int {
		return fset.File(p).Offset(p)
	}

	srcName := path.Base(srcMod)
	dstName := path.Base(dstMod)
	if isRoot {
		if name := f.Name.Name; name == srcName || name == srcName+"_test" {
			dname := dstName + strings.TrimPrefix(name, srcName)
			if !token.IsIdentifier(dname) {
				log.Fatalf("%s: cannot rename package %s to package %s: invalid package name", file, name, dname)
			}
			buf.Replace(at(f.Name.Pos()), at(f.Name.End()), dname)
		}
	}

	for _, spec := range f.Imports {
		path, err := strconv.Unquote(spec.Path.Value)
		if err != nil {
			continue
		}
		if path == srcMod {
			if srcName != dstName && spec.Name == nil {
				// Add package rename because source code uses original name.
				// The renaming looks strange, but template authors are unlikely to
				// create a template where the root package is imported by packages
				// in subdirectories, and the renaming at least keeps the code working.
				// A more sophisticated approach would be to rename the uses of
				// the package identifier in the file too, but then you have to worry about
				// name collisions, and given how unlikely this is, it doesn't seem worth
				// trying to clean up the file that way.
				buf.Insert(at(spec.Path.Pos()), srcName+" ")
			}
			// Change import path to dstMod
			buf.Replace(at(spec.Path.Pos()), at(spec.Path.End()), strconv.Quote(dstMod))
		}
		if strings.HasPrefix(path, srcMod+"/") {
			// Change import path to begin with dstMod
			buf.Replace(at(spec.Path.Pos()), at(spec.Path.End()), strconv.Quote(strings.Replace(path, srcMod, dstMod, 1)))
		}
	}
	return buf.Bytes()
}

// fixGoMod rewrites the go.mod content in data to replace srcMod with dstMod
// in the module path.
func fixGoMod(data []byte, srcMod, dstMod string) []byte {
	f, err := modfile.ParseLax("go.mod", data, nil)
	if err != nil {
		log.Fatalf("parsing source module:\n%s", err)
	}
	f.AddModuleStmt(dstMod)
	new, err := f.Format()
	if err != nil {
		return data
	}
	return new
}
