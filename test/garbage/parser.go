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
	"os"
	"path"
	"runtime"
	"strings"
)

func isGoFile(dir *os.Dir) bool {
	return dir.IsRegular() &&
		!strings.HasPrefix(dir.Name, ".") && // ignore .files
		path.Ext(dir.Name) == ".go"
}

func isPkgFile(dir *os.Dir) bool {
	return isGoFile(dir) &&
		!strings.HasSuffix(dir.Name, "_test.go") // ignore test files
}

func pkgName(filename string) string {
	file, err := parser.ParseFile(filename, nil, nil, parser.PackageClauseOnly)
	if err != nil || file == nil {
		return ""
	}
	return file.Name.Name()
}

func parseDir(dirpath string) map[string]*ast.Package {
	// the package name is the directory name within its parent
	// (use dirname instead of path because dirname is clean; i.e. has no trailing '/')
	_, pkgname := path.Split(dirpath)

	// filter function to select the desired .go files
	filter := func(d *os.Dir) bool {
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
	pkgs, err := parser.ParseDir(dirpath, filter, parser.ParseComments)
	if err != nil {
		panicln("parse", dirpath, err.String())
	}
	return pkgs
}

func main() {
	st := &runtime.MemStats
	n := flag.Int("n", 10, "iterations")
	p := flag.Int("p", len(packages), "# of packages to keep in memory")
	flag.BoolVar(&st.DebugGC, "d", st.DebugGC, "print GC debugging info (pause times)")
	flag.Parse()

	pkgroot := os.Getenv("GOROOT") + "/src/pkg/"
	for i := -1; i < *n; i++ {
		parsed := make([]map[string]*ast.Package, *p)
		for j := range parsed {
			parsed[j] = parseDir(pkgroot + packages[j%len(packages)])
		}
		if i == -1 {
			// Now that heap is grown to full size, reset counters.
			// This hides the start-up pauses, which are much smaller
			// than the normal pauses and would otherwise make
			// the average look much better than it actually is.
			st.NumGC = 0
			st.PauseNs = 0
		}
	}

	fmt.Printf("Alloc=%d/%d Heap=%d/%d Mallocs=%d PauseTime=%.3f/%d = %.3f\n",
		st.Alloc, st.TotalAlloc,
		st.InusePages<<12, st.Sys,
		st.Mallocs, float64(st.PauseNs)/1e9,
		st.NumGC, float64(st.PauseNs)/1e9/float64(st.NumGC))

	fmt.Printf("%10s %10s %10s\n", "size", "#alloc", "#free")
	for _, s := range st.BySize {
		fmt.Printf("%10d %10d %10d\n", s.Size, s.Mallocs, s.Frees)
	}
}


var packages = []string{
	"archive/tar",
	"asn1",
	"big",
	"bignum",
	"bufio",
	"bytes",
	"compress/flate",
	"compress/gzip",
	"compress/zlib",
	"container/heap",
	"container/list",
	"container/ring",
	"container/vector",
	"crypto/aes",
	"crypto/block",
	"crypto/hmac",
	"crypto/md4",
	"crypto/md5",
	"crypto/rc4",
	"crypto/rsa",
	"crypto/sha1",
	"crypto/sha256",
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
	"exp/exception",
	"exp/iterable",
	"exp/parser",
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
	"http",
	"image",
	"image/jpeg",
	"image/png",
	"io",
	"io/ioutil",
	"json",
	"log",
	"math",
	"net",
	"once",
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
	"websocket",
	"xgb",
	"xml",
}
