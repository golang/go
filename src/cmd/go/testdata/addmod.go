// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

// Addmod adds a module as a txtar archive to the testdata/mod directory.
//
// Usage:
//
//	go run addmod.go path@version...
//
// It should only be used for very small modules - we do not want to check
// very large files into testdata/mod.
//
// It is acceptable to edit the archive afterward to remove or shorten files.
// See mod/README for more information.
//
package main

import (
	"bytes"
	"flag"
	"fmt"
	exec "internal/execabs"
	"internal/txtar"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: go run addmod.go path@version...\n")
	os.Exit(2)
}

var tmpdir string

func fatalf(format string, args ...any) {
	os.RemoveAll(tmpdir)
	log.Fatalf(format, args...)
}

const goCmd = "go"

func main() {
	flag.Usage = usage
	flag.Parse()
	if flag.NArg() == 0 {
		usage()
	}

	log.SetPrefix("addmod: ")
	log.SetFlags(0)

	var err error
	tmpdir, err = os.MkdirTemp("", "addmod-")
	if err != nil {
		log.Fatal(err)
	}

	run := func(command string, args ...string) string {
		cmd := exec.Command(command, args...)
		cmd.Dir = tmpdir
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		out, err := cmd.Output()
		if err != nil {
			fatalf("%s %s: %v\n%s", command, strings.Join(args, " "), err, stderr.Bytes())
		}
		return string(out)
	}

	gopath := strings.TrimSpace(run("go", "env", "GOPATH"))
	if gopath == "" {
		fatalf("cannot find GOPATH")
	}

	exitCode := 0
	for _, arg := range flag.Args() {
		if err := os.WriteFile(filepath.Join(tmpdir, "go.mod"), []byte("module m\n"), 0666); err != nil {
			fatalf("%v", err)
		}
		run(goCmd, "get", "-d", arg)
		path := arg
		if i := strings.Index(path, "@"); i >= 0 {
			path = path[:i]
		}
		out := run(goCmd, "list", "-m", "-f={{.Path}} {{.Version}} {{.Dir}}", path)
		f := strings.Fields(out)
		if len(f) != 3 {
			log.Printf("go list -m %s: unexpected output %q", arg, out)
			exitCode = 1
			continue
		}
		path, vers, dir := f[0], f[1], f[2]
		mod, err := os.ReadFile(filepath.Join(gopath, "pkg/mod/cache/download", path, "@v", vers+".mod"))
		if err != nil {
			log.Printf("%s: %v", arg, err)
			exitCode = 1
			continue
		}
		info, err := os.ReadFile(filepath.Join(gopath, "pkg/mod/cache/download", path, "@v", vers+".info"))
		if err != nil {
			log.Printf("%s: %v", arg, err)
			exitCode = 1
			continue
		}

		a := new(txtar.Archive)
		title := arg
		if !strings.Contains(arg, "@") {
			title += "@" + vers
		}
		a.Comment = []byte(fmt.Sprintf("module %s\n\n", title))
		a.Files = []txtar.File{
			{Name: ".mod", Data: mod},
			{Name: ".info", Data: info},
		}
		dir = filepath.Clean(dir)
		err = filepath.WalkDir(dir, func(path string, info fs.DirEntry, err error) error {
			if !info.Type().IsRegular() {
				return nil
			}
			name := info.Name()
			if name == "go.mod" || strings.HasSuffix(name, ".go") {
				data, err := os.ReadFile(path)
				if err != nil {
					return err
				}
				a.Files = append(a.Files, txtar.File{Name: strings.TrimPrefix(path, dir+string(filepath.Separator)), Data: data})
			}
			return nil
		})
		if err != nil {
			log.Printf("%s: %v", arg, err)
			exitCode = 1
			continue
		}

		data := txtar.Format(a)
		target := filepath.Join("mod", strings.ReplaceAll(path, "/", "_")+"_"+vers+".txt")
		if err := os.WriteFile(target, data, 0666); err != nil {
			log.Printf("%s: %v", arg, err)
			exitCode = 1
			continue
		}
	}
	os.RemoveAll(tmpdir)
	os.Exit(exitCode)
}
