// Copyright 2010 The Go Authors. All rights reserved.
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
	"log"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path"
	"runtime"
	"strings"
	"time"
)

var serve = flag.String("serve", "", "serve http on this address at end")

func isGoFile(dir os.FileInfo) bool {
	return !dir.IsDir() &&
		!strings.HasPrefix(dir.Name(), ".") && // ignore .files
		path.Ext(dir.Name()) == ".go"
}

func isPkgFile(dir os.FileInfo) bool {
	return isGoFile(dir) &&
		!strings.HasSuffix(dir.Name(), "_test.go") // ignore test files
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
	filter := func(d os.FileInfo) bool {
		if isPkgFile(d) {
			// Some directories contain main packages: Only accept
			// files that belong to the expected package so that
			// parser.ParsePackage doesn't return "multiple packages
			// found" errors.
			// Additionally, accept the special package name
			// fakePkgName if we are looking at cmd documentation.
			name := pkgName(dirpath + "/" + d.Name())
			return name == pkgname
		}
		return false
	}

	// get package AST
	pkgs, err := parser.ParseDir(token.NewFileSet(), dirpath, filter, parser.ParseComments)
	if err != nil {
		println("parse", dirpath, err.Error())
		panic("fail")
	}
	return pkgs
}

func main() {
	st := new(runtime.MemStats)
	packages = append(packages, packages...)
	packages = append(packages, packages...)
	n := flag.Int("n", 4, "iterations")
	p := flag.Int("p", len(packages), "# of packages to keep in memory")
	flag.BoolVar(&st.DebugGC, "d", st.DebugGC, "print GC debugging info (pause times)")
	flag.Parse()

	var lastParsed []map[string]*ast.Package
	var t0 time.Time
	var numGC uint32
	var pauseTotalNs uint64
	pkgroot := runtime.GOROOT() + "/src/"
	for pass := 0; pass < 2; pass++ {
		// Once the heap is grown to full size, reset counters.
		// This hides the start-up pauses, which are much smaller
		// than the normal pauses and would otherwise make
		// the average look much better than it actually is.
		runtime.ReadMemStats(st)
		numGC = st.NumGC
		pauseTotalNs = st.PauseTotalNs
		t0 = time.Now()

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
	t1 := time.Now()

	runtime.ReadMemStats(st)
	st.NumGC -= numGC
	st.PauseTotalNs -= pauseTotalNs
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
	gcstats("BenchmarkParser", *n, t1.Sub(t0))

	if *serve != "" {
		log.Fatal(http.ListenAndServe(*serve, nil))
		println(lastParsed)
	}
}

// find . -type d -not -path "./exp" -not -path "./exp/*" -printf "\t\"%p\",\n" | sort | sed "s/\.\///" | grep -v testdata
var packages = []string{
	"archive",
	"archive/tar",
	"archive/zip",
	"bufio",
	"builtin",
	"bytes",
	"compress",
	"compress/bzip2",
	"compress/flate",
	"compress/gzip",
	"compress/lzw",
	"compress/zlib",
	"container",
	"container/heap",
	"container/list",
	"container/ring",
	"crypto",
	"crypto/aes",
	"crypto/cipher",
	"crypto/des",
	"crypto/dsa",
	"crypto/ecdsa",
	"crypto/elliptic",
	"crypto/hmac",
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
	"crypto/x509/pkix",
	"database",
	"database/sql",
	"database/sql/driver",
	"debug",
	"debug/dwarf",
	"debug/elf",
	"debug/gosym",
	"debug/macho",
	"debug/pe",
	"encoding",
	"encoding/ascii85",
	"encoding/asn1",
	"encoding/base32",
	"encoding/base64",
	"encoding/binary",
	"encoding/csv",
	"encoding/gob",
	"encoding/hex",
	"encoding/json",
	"encoding/pem",
	"encoding/xml",
	"errors",
	"expvar",
	"flag",
	"fmt",
	"go",
	"go/ast",
	"go/build",
	"go/doc",
	"go/format",
	"go/parser",
	"go/printer",
	"go/scanner",
	"go/token",
	"hash",
	"hash/adler32",
	"hash/crc32",
	"hash/crc64",
	"hash/fnv",
	"html",
	"html/template",
	"image",
	"image/color",
	"image/draw",
	"image/gif",
	"image/jpeg",
	"image/png",
	"index",
	"index/suffixarray",
	"io",
	"io/ioutil",
	"log",
	"log/syslog",
	"math",
	"math/big",
	"math/cmplx",
	"math/rand",
	"mime",
	"mime/multipart",
	"net",
	"net/http",
	"net/http/cgi",
	"net/http/cookiejar",
	"net/http/fcgi",
	"net/http/httptest",
	"net/http/httputil",
	"net/http/pprof",
	"net/mail",
	"net/rpc",
	"net/rpc/jsonrpc",
	"net/smtp",
	"net/textproto",
	"net/url",
	"os",
	"os/exec",
	"os/signal",
	"os/user",
	"path",
	"path/filepath",
	"reflect",
	"regexp",
	"regexp/syntax",
	"runtime",
	"runtime/cgo",
	"runtime/debug",
	"runtime/pprof",
	"runtime/race",
	"sort",
	"strconv",
	"strings",
	"sync",
	"sync/atomic",
	"syscall",
	"testing",
	"testing/iotest",
	"testing/quick",
	"text",
	"text/scanner",
	"text/tabwriter",
	"text/template",
	"text/template/parse",
	"time",
	"unicode",
	"unicode/utf16",
	"unicode/utf8",
	"unsafe",
}
