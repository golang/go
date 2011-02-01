// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collection benchmark: parse Go packages repeatedly.

package main

import (
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path"
	"runtime"
	"strings"
	"time"
	"http"
	_ "http/pprof"
	"log"
)

var serve = flag.String("serve", "", "serve http on this address at end")

func isGoFile(dir *os.FileInfo) bool {
	return dir.IsRegular() &&
		!strings.HasPrefix(dir.Name, ".") && // ignore .files
		path.Ext(dir.Name) == ".go"
}

func isPkgFile(dir *os.FileInfo) bool {
	return isGoFile(dir) &&
		!strings.HasSuffix(dir.Name, "_test.go") // ignore test files
}

func pkgName(filename string) string {
	file, err := parser.ParseFile(token.NewFileSet(), filename, nil, parser.PackageClauseOnly)
	if err != nil || file == nil {
		return ""
	}
	return file.Name.Name
}

func parseDir(dirpath string) map[string]*ast.Package {
	// the package name is the directory name within its parent
	// (use dirname instead of path because dirname is clean; i.e. has no trailing '/')
	_, pkgname := path.Split(dirpath)

	// filter function to select the desired .go files
	filter := func(d *os.FileInfo) bool {
		if isPkgFile(d) {
			// Some directories contain main packages: Only accept
			// files that belong to the expected package so that
			// parser.ParsePackage doesn't return "multiple packages
			// found" errors.
			// Additionally, accept the special package name
			// fakePkgName if we are looking at cmd documentation.
			name := pkgName(dirpath + "/" + d.Name)
			return name == pkgname
		}
		return false
	}

	// get package AST
	pkgs, err := parser.ParseDir(token.NewFileSet(), dirpath, filter, parser.ParseComments)
	if err != nil {
		println("parse", dirpath, err.String())
		panic("fail")
	}
	return pkgs
}

func main() {
	runtime.GOMAXPROCS(4)
	go func() {}()
	go func() {}()
	go func() {}()
	st := &runtime.MemStats
	packages = append(packages, packages...)
	packages = append(packages, packages...)
	n := flag.Int("n", 4, "iterations")
	p := flag.Int("p", len(packages), "# of packages to keep in memory")
	flag.BoolVar(&st.DebugGC, "d", st.DebugGC, "print GC debugging info (pause times)")
	flag.Parse()

	var lastParsed []map[string]*ast.Package
	var t0 int64
	pkgroot := runtime.GOROOT() + "/src/pkg/"
	for pass := 0; pass < 2; pass++ {
		// Once the heap is grown to full size, reset counters.
		// This hides the start-up pauses, which are much smaller
		// than the normal pauses and would otherwise make
		// the average look much better than it actually is.
		st.NumGC = 0
		st.PauseTotalNs = 0
		t0 = time.Nanoseconds()

		for i := 0; i < *n; i++ {
			parsed := make([]map[string]*ast.Package, *p)
			for j := range parsed {
				parsed[j] = parseDir(pkgroot + packages[j%len(packages)])
			}
			if i+1 == *n && *serve != "" {
				lastParsed = parsed
			}
		}
		runtime.GC()
		runtime.GC()
	}
	t1 := time.Nanoseconds()

	fmt.Printf("Alloc=%d/%d Heap=%d Mallocs=%d PauseTime=%.3f/%d = %.3f\n",
		st.Alloc, st.TotalAlloc,
		st.Sys,
		st.Mallocs, float64(st.PauseTotalNs)/1e9,
		st.NumGC, float64(st.PauseTotalNs)/1e9/float64(st.NumGC))

	/*
		fmt.Printf("%10s %10s %10s\n", "size", "#alloc", "#free")
		for _, s := range st.BySize {
			fmt.Printf("%10d %10d %10d\n", s.Size, s.Mallocs, s.Frees)
		}
	*/
	// Standard gotest benchmark output, collected by build dashboard.
	gcstats("BenchmarkParser", *n, t1-t0)

	if *serve != "" {
		log.Fatal(http.ListenAndServe(*serve, nil))
		println(lastParsed)
	}
}


var packages = []string{
	"archive/tar",
	"asn1",
	"big",
	"bufio",
	"bytes",
	"cmath",
	"compress/flate",
	"compress/gzip",
	"compress/zlib",
	"container/heap",
	"container/list",
	"container/ring",
	"container/vector",
	"crypto/aes",
	"crypto/block",
	"crypto/blowfish",
	"crypto/hmac",
	"crypto/md4",
	"crypto/md5",
	"crypto/rand",
	"crypto/rc4",
	"crypto/rsa",
	"crypto/sha1",
	"crypto/sha256",
	"crypto/sha512",
	"crypto/subtle",
	"crypto/tls",
	"crypto/x509",
	"crypto/xtea",
	"debug/dwarf",
	"debug/macho",
	"debug/elf",
	"debug/gosym",
	"debug/proc",
	"ebnf",
	"encoding/ascii85",
	"encoding/base64",
	"encoding/binary",
	"encoding/git85",
	"encoding/hex",
	"encoding/pem",
	"exec",
	"exp/datafmt",
	"exp/draw",
	"exp/eval",
	"exp/iterable",
	"expvar",
	"flag",
	"fmt",
	"go/ast",
	"go/doc",
	"go/parser",
	"go/printer",
	"go/scanner",
	"go/token",
	"gob",
	"hash",
	"hash/adler32",
	"hash/crc32",
	"hash/crc64",
	"http",
	"image",
	"image/jpeg",
	"image/png",
	"io",
	"io/ioutil",
	"json",
	"log",
	"math",
	"mime",
	"net",
	"os",
	"os/signal",
	"patch",
	"path",
	"rand",
	"reflect",
	"regexp",
	"rpc",
	"runtime",
	"scanner",
	"sort",
	"smtp",
	"strconv",
	"strings",
	"sync",
	"syscall",
	"syslog",
	"tabwriter",
	"template",
	"testing",
	"testing/iotest",
	"testing/quick",
	"testing/script",
	"time",
	"unicode",
	"utf8",
	"utf16",
	"websocket",
	"xml",
}
