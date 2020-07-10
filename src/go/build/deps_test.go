// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file exercises the import parser but also checks that
// some low-level packages do not have new dependencies added.

package build

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// depsRules defines the expected dependencies between packages in
// the Go source tree. It is a statement of policy.
//
// DO NOT CHANGE THIS DATA TO FIX BUILDS.
// Existing packages should not have their constraints relaxed
// without prior discussion.
// Negative assertions should almost never be removed.
//
// The general syntax of a rule is:
//
//		a, b < c, d;
//
// which means c and d come after a and b in the partial order
// (that is, c and d can import a and b),
// but doesn't provide a relative order between a vs b or c vs d.
//
// The rules can chain together, as in:
//
//		e < f, g < h;
//
// which is equivalent to
//
//		e < f, g;
//		f, g < h;
//
// Except for the special bottom element "NONE", each name
// must appear exactly once on the right-hand side of a rule.
// That rule serves as the definition of the allowed dependencies
// for that name. The definition must appear before any uses
// of the name on the left-hand side of a rule. (That is, the
// rules themselves must be ordered according to the partial
// order, for easier reading by people.)
//
// Negative assertions double-check the partial order:
//
//		i !< j
//
// means that it must NOT be the case that i < j.
// Negative assertions may appear anywhere in the rules,
// even before i and j have been defined.
//
// Comments begin with #.
//
// All-caps names are pseudo-names for specific points
// in the dependency lattice.
//
var depsRules = `
	# No dependencies allowed for any of these packages.
	NONE
	< container/list, container/ring,
	  internal/cfg, internal/cpu,
	  internal/goversion, internal/nettrace,
	  unicode/utf8, unicode/utf16, unicode,
	  unsafe;

	# RUNTIME is the core runtime group of packages, all of them very light-weight.
	internal/cpu, unsafe
	< internal/bytealg
	< internal/unsafeheader
	< runtime/internal/sys
	< runtime/internal/atomic
	< runtime/internal/math
	< runtime
	< sync/atomic
	< internal/race
	< sync
	< internal/reflectlite
	< errors
	< internal/oserror, math/bits
	< RUNTIME;

	RUNTIME
	< sort
	< container/heap;

	RUNTIME
	< io;

	reflect !< sort;

	# SYSCALL is RUNTIME plus the packages necessary for basic system calls.
	RUNTIME, unicode/utf8, unicode/utf16, io
	< internal/syscall/windows/sysdll, syscall/js
	< syscall
	< internal/syscall/unix, internal/syscall/windows, internal/syscall/windows/registry
	< internal/syscall/execenv
	< SYSCALL;

	# TIME is SYSCALL plus the core packages about time, including context.
	SYSCALL
	< time/tzdata
	< time
	< context
	< TIME;

	# MATH is RUNTIME plus the basic math packages.
	RUNTIME
	< math
	< MATH;

	unicode !< math;

	MATH
	< math/cmplx;

	MATH
	< math/rand;

	MATH, unicode/utf8
	< strconv;

	unicode !< strconv;

	# STR is basic string and buffer manipulation.
	RUNTIME, io, unicode/utf8, unicode/utf16, unicode
	< bytes, strings
	< bufio, path;

	bufio, path, strconv
	< STR;

	# OS is basic OS access, including helpers (path/filepath, os/exec, etc).
	# OS includes string routines, but those must be layered above package os.
	# OS does not include reflection.
	TIME, io, sort
	< internal/testlog
	< internal/poll
	< os
	< os/signal;

	unicode, fmt !< os, os/signal;

	os/signal, STR
	< path/filepath
	< io/ioutil, os/exec
	< OS;

	reflect !< OS;

	OS
	< golang.org/x/sys/cpu, internal/goroot;

	# FMT is OS (which includes string routines) plus reflect and fmt.
	# It does not include package log, which should be avoided in core packages.
	strconv, unicode
	< reflect;

	os, reflect
	< internal/fmtsort
	< fmt;

	OS, fmt
	< FMT;

	log !< FMT;

	# Misc packages needing only FMT.
	FMT
	< flag,
	  html,
	  mime/quotedprintable,
	  net/internal/socktest,
	  net/url,
	  runtime/debug,
	  runtime/trace,
	  text/scanner,
	  text/tabwriter;

	# encodings
	# core ones do not use fmt.
	io, strconv
	< encoding;

	encoding, reflect
	< encoding/binary
	< encoding/base32, encoding/base64;

	fmt !< encoding/base32, encoding/base64;

	FMT, encoding/base32, encoding/base64
	< encoding/ascii85, encoding/csv, encoding/gob, encoding/hex,
	  encoding/json, encoding/pem, encoding/xml, mime;

	# hashes
	io
	< hash
	< hash/adler32, hash/crc32, hash/crc64, hash/fnv, hash/maphash;

	# math/big
	FMT, encoding/binary, math/rand
	< math/big;

	# compression
	FMT, encoding/binary, hash/adler32, hash/crc32
	< compress/bzip2, compress/flate, compress/lzw
	< archive/zip, compress/gzip, compress/zlib;

	# templates
	FMT
	< text/template/parse;

	net/url, text/template/parse
	< text/template
	< internal/lazytemplate;

	encoding/json, html, text/template
	< html/template;

	# regexp
	FMT
	< regexp/syntax
	< regexp
	< internal/lazyregexp;

	# suffix array
	encoding/binary, regexp
	< index/suffixarray;

	# executable parsing
	FMT, encoding/binary, compress/zlib
	< debug/dwarf
	< debug/elf, debug/gosym, debug/macho, debug/pe, debug/plan9obj, internal/xcoff
	< DEBUG;

	# go parser and friends.
	FMT
	< go/token
	< go/scanner
	< go/ast
	< go/parser;

	go/parser, text/tabwriter
	< go/printer
	< go/format;

	go/parser, internal/lazyregexp, text/template
	< go/doc;

	math/big, go/token
	< go/constant;

	container/heap, go/constant, go/parser
	< go/types;

	go/doc, go/parser, internal/goroot, internal/goversion
	< go/build;

	DEBUG, go/build, go/types, text/scanner
	< go/internal/gcimporter, go/internal/gccgoimporter, go/internal/srcimporter
	< go/importer;

	# databases
	FMT
	< database/sql/internal
	< database/sql/driver
	< database/sql;

	# images
	FMT, compress/lzw, compress/zlib
	< image/color
	< image, image/color/palette
	< image/internal/imageutil
	< image/draw
	< image/gif, image/jpeg, image/png;

	# cgo, delayed as long as possible.
	# If you add a dependency on CGO, you must add the package
	# to cgoPackages in cmd/dist/test.go as well.
	RUNTIME
	< C
	< runtime/cgo
	< CGO
	< runtime/race, runtime/msan;

	# Bulk of the standard library must not use cgo.
	# The prohibition stops at net and os/user.
	C !< fmt, go/types, CRYPTO-MATH;

	CGO, OS
	< plugin;

	CGO, FMT
	< os/user
	< archive/tar;

	sync
	< internal/singleflight;

	os
	< golang.org/x/net/dns/dnsmessage,
	  golang.org/x/net/lif,
	  golang.org/x/net/route;

	# net is unavoidable when doing any networking,
	# so large dependencies must be kept out.
	# This is a long-looking list but most of these
	# are small with few dependencies.
	# math/rand should probably be removed at some point.
	CGO,
	golang.org/x/net/dns/dnsmessage,
	golang.org/x/net/lif,
	golang.org/x/net/route,
	internal/nettrace,
	internal/poll,
	internal/singleflight,
	internal/race,
	math/rand,
	os
	< net;

	fmt, unicode !< net;

	# NET is net plus net-helper packages.
	FMT, net
	< net/textproto;

	mime, net/textproto, net/url
	< NET;

	# logging - most packages should not import; http and up is allowed
	FMT
	< log;

	log !< crypto/tls, database/sql, go/importer, testing;

	FMT, log, net
	< log/syslog;

	NET, log
	< net/mail;

	# CRYPTO is core crypto algorithms - no cgo, fmt, net.
	# Unfortunately, stuck with reflect via encoding/binary.
	encoding/binary, golang.org/x/sys/cpu, hash
	< crypto
	< crypto/subtle
	< crypto/internal/subtle
	< crypto/cipher
	< crypto/aes, crypto/des, crypto/hmac, crypto/md5, crypto/rc4,
	  crypto/sha1, crypto/sha256, crypto/sha512
	< CRYPTO;

	CGO, fmt, net !< CRYPTO;

	# CRYPTO-MATH is core bignum-based crypto - no cgo, net; fmt now ok.
	CRYPTO, FMT, math/big
	< crypto/rand
	< crypto/internal/randutil
	< crypto/ed25519/internal/edwards25519
	< crypto/ed25519
	< encoding/asn1
	< golang.org/x/crypto/cryptobyte/asn1
	< golang.org/x/crypto/cryptobyte
	< golang.org/x/crypto/curve25519
	< crypto/dsa, crypto/elliptic, crypto/rsa
	< crypto/ecdsa
	< CRYPTO-MATH;

	CGO, net !< CRYPTO-MATH;

	# TLS, Prince of Dependencies.
	CGO, CRYPTO-MATH, NET, container/list, encoding/hex, encoding/pem
	< golang.org/x/crypto/internal/subtle
	< golang.org/x/crypto/chacha20
	< golang.org/x/crypto/poly1305
	< golang.org/x/crypto/chacha20poly1305
	< golang.org/x/crypto/hkdf
	< crypto/x509/internal/macos
	< crypto/x509/pkix
	< crypto/x509
	< crypto/tls;

	# crypto-aware packages

	NET, crypto/rand, mime/quotedprintable
	< mime/multipart;

	crypto/tls
	< net/smtp;

	# HTTP, King of Dependencies.

	FMT
	< golang.org/x/net/http2/hpack, net/http/internal;

	FMT, NET, container/list, encoding/binary, log
	< golang.org/x/text/transform
	< golang.org/x/text/unicode/norm
	< golang.org/x/text/unicode/bidi
	< golang.org/x/text/secure/bidirule
	< golang.org/x/net/idna
	< golang.org/x/net/http/httpguts, golang.org/x/net/http/httpproxy;

	NET, crypto/tls
	< net/http/httptrace;

	compress/gzip,
	golang.org/x/net/http/httpguts,
	golang.org/x/net/http/httpproxy,
	golang.org/x/net/http2/hpack,
	net/http/internal,
	net/http/httptrace,
	mime/multipart,
	log
	< net/http;

	# HTTP-aware packages

	encoding/json, net/http
	< expvar;

	net/http
	< net/http/cookiejar, net/http/httputil;

	net/http, flag
	< net/http/httptest;

	net/http, regexp
	< net/http/cgi
	< net/http/fcgi;

	# Profiling
	FMT, compress/gzip, encoding/binary, text/tabwriter
	< runtime/pprof;

	OS, compress/gzip, regexp
	< internal/profile;

	html/template, internal/profile, net/http, runtime/pprof, runtime/trace
	< net/http/pprof;

	# RPC
	encoding/gob, encoding/json, go/token, html/template, net/http
	< net/rpc
	< net/rpc/jsonrpc;

	# Test-only
	log
	< testing/iotest;

	FMT, flag, math/rand
	< testing/quick;

	FMT, flag, runtime/debug, runtime/trace
	< testing;

	internal/testlog, runtime/pprof, regexp
	< testing/internal/testdeps;

	OS, flag, testing, internal/cfg
	< internal/testenv;

	OS, encoding/base64
	< internal/obscuretestdata;

	CGO, OS, fmt
	< os/signal/internal/pty;

	NET, testing
	< golang.org/x/net/nettest;

	FMT, container/heap, math/rand
	< internal/trace;
`

// listStdPkgs returns the same list of packages as "go list std".
func listStdPkgs(goroot string) ([]string, error) {
	// Based on cmd/go's matchPackages function.
	var pkgs []string

	src := filepath.Join(goroot, "src") + string(filepath.Separator)
	walkFn := func(path string, fi os.FileInfo, err error) error {
		if err != nil || !fi.IsDir() || path == src {
			return nil
		}

		base := filepath.Base(path)
		if strings.HasPrefix(base, ".") || strings.HasPrefix(base, "_") || base == "testdata" {
			return filepath.SkipDir
		}

		name := filepath.ToSlash(path[len(src):])
		if name == "builtin" || name == "cmd" {
			return filepath.SkipDir
		}

		pkgs = append(pkgs, strings.TrimPrefix(name, "vendor/"))
		return nil
	}
	if err := filepath.Walk(src, walkFn); err != nil {
		return nil, err
	}
	return pkgs, nil
}

func TestDependencies(t *testing.T) {
	if !testenv.HasSrc() {
		// Tests run in a limited file system and we do not
		// provide access to every source file.
		t.Skipf("skipping on %s/%s, missing full GOROOT", runtime.GOOS, runtime.GOARCH)
	}

	ctxt := Default
	all, err := listStdPkgs(ctxt.GOROOT)
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(all)

	sawImport := map[string]map[string]bool{} // from package => to package => true
	policy := depsPolicy(t)

	for _, pkg := range all {
		imports, err := findImports(pkg)
		if err != nil {
			t.Error(err)
			continue
		}
		if sawImport[pkg] == nil {
			sawImport[pkg] = map[string]bool{}
		}
		ok := policy[pkg]
		var bad []string
		for _, imp := range imports {
			sawImport[pkg][imp] = true
			if !ok[imp] {
				bad = append(bad, imp)
			}
		}
		if bad != nil {
			t.Errorf("unexpected dependency: %s imports %v", pkg, bad)
		}
	}

	// depPath returns the path between the given from and to packages.
	// It returns the empty string if there's no dependency path.
	var depPath func(string, string) string
	depPath = func(from, to string) string {
		if sawImport[from][to] {
			return from + " => " + to
		}
		for pkg := range sawImport[from] {
			if p := depPath(pkg, to); p != "" {
				return from + " => " + p
			}
		}
		return ""
	}
}

var buildIgnore = []byte("\n// +build ignore")

func findImports(pkg string) ([]string, error) {
	vpkg := pkg
	if strings.HasPrefix(pkg, "golang.org") {
		vpkg = "vendor/" + pkg
	}
	dir := filepath.Join(Default.GOROOT, "src", vpkg)
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var imports []string
	var haveImport = map[string]bool{}
	for _, file := range files {
		name := file.Name()
		if name == "slice_go14.go" || name == "slice_go18.go" {
			// These files are for compiler bootstrap with older versions of Go and not built in the standard build.
			continue
		}
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		f, err := os.Open(filepath.Join(dir, name))
		if err != nil {
			return nil, err
		}
		var imp []string
		data, err := readImports(f, false, &imp)
		f.Close()
		if err != nil {
			return nil, fmt.Errorf("reading %v: %v", name, err)
		}
		if bytes.Contains(data, buildIgnore) {
			continue
		}
		for _, quoted := range imp {
			path, err := strconv.Unquote(quoted)
			if err != nil {
				continue
			}
			if !haveImport[path] {
				haveImport[path] = true
				imports = append(imports, path)
			}
		}
	}
	sort.Strings(imports)
	return imports, nil
}

// depsPolicy returns a map m such that m[p][d] == true when p can import d.
func depsPolicy(t *testing.T) map[string]map[string]bool {
	allowed := map[string]map[string]bool{"NONE": {}}
	disallowed := [][2][]string{}

	parseDepsRules(t, func(deps []string, op string, users []string) {
		if op == "!<" {
			disallowed = append(disallowed, [2][]string{deps, users})
			return
		}
		for _, u := range users {
			if allowed[u] != nil {
				t.Errorf("multiple deps lists for %s", u)
			}
			allowed[u] = make(map[string]bool)
			for _, d := range deps {
				if allowed[d] == nil {
					t.Errorf("use of %s before its deps list", d)
				}
				allowed[u][d] = true
			}
		}
	})

	// Check for missing deps info.
	for _, deps := range allowed {
		for d := range deps {
			if allowed[d] == nil {
				t.Errorf("missing deps list for %s", d)
			}
		}
	}

	// Complete transitive allowed deps.
	for k := range allowed {
		for i := range allowed {
			for j := range allowed {
				if i != k && k != j && allowed[i][k] && allowed[k][j] {
					if i == j {
						// Can only happen along with a "use of X before deps" error above,
						// but this error is more specific - it makes clear that reordering the
						// rules will not be enough to fix the problem.
						t.Errorf("deps policy cycle: %s < %s < %s", j, k, i)
					}
					allowed[i][j] = true
				}
			}
		}
	}

	// Check negative assertions against completed allowed deps.
	for _, bad := range disallowed {
		deps, users := bad[0], bad[1]
		for _, d := range deps {
			for _, u := range users {
				if allowed[u][d] {
					t.Errorf("deps policy incorrect: assertion failed: %s !< %s", d, u)
				}
			}
		}
	}

	if t.Failed() {
		t.FailNow()
	}

	return allowed
}

// parseDepsRules parses depsRules, calling save(deps, op, users)
// for each deps < users or deps !< users rule
// (op is "<" or "!<").
func parseDepsRules(t *testing.T, save func(deps []string, op string, users []string)) {
	p := &depsParser{t: t, lineno: 1, text: depsRules}

	var prev []string
	var op string
	for {
		list, tok := p.nextList()
		if tok == "" {
			if prev == nil {
				break
			}
			p.syntaxError("unexpected EOF")
		}
		if prev != nil {
			save(prev, op, list)
		}
		prev = list
		if tok == ";" {
			prev = nil
			op = ""
			continue
		}
		if tok != "<" && tok != "!<" {
			p.syntaxError("missing <")
		}
		op = tok
	}
}

// A depsParser parses the depsRules syntax described above.
type depsParser struct {
	t        *testing.T
	lineno   int
	lastWord string
	text     string
}

// syntaxError reports a parsing error.
func (p *depsParser) syntaxError(msg string) {
	p.t.Fatalf("deps:%d: syntax error: %s near %s", p.lineno, msg, p.lastWord)
}

// nextList parses and returns a comma-separated list of names.
func (p *depsParser) nextList() (list []string, token string) {
	for {
		tok := p.nextToken()
		switch tok {
		case "":
			if len(list) == 0 {
				return nil, ""
			}
			fallthrough
		case ",", "<", "!<", ";":
			p.syntaxError("bad list syntax")
		}
		list = append(list, tok)

		tok = p.nextToken()
		if tok != "," {
			return list, tok
		}
	}
}

// nextToken returns the next token in the deps rules,
// one of ";" "," "<" "!<" or a name.
func (p *depsParser) nextToken() string {
	for {
		if p.text == "" {
			return ""
		}
		switch p.text[0] {
		case ';', ',', '<':
			t := p.text[:1]
			p.text = p.text[1:]
			return t

		case '!':
			if len(p.text) < 2 || p.text[1] != '<' {
				p.syntaxError("unexpected token !")
			}
			p.text = p.text[2:]
			return "!<"

		case '#':
			i := strings.Index(p.text, "\n")
			if i < 0 {
				i = len(p.text)
			}
			p.text = p.text[i:]
			continue

		case '\n':
			p.lineno++
			fallthrough
		case ' ', '\t':
			p.text = p.text[1:]
			continue

		default:
			i := strings.IndexAny(p.text, "!;,<#\n \t")
			if i < 0 {
				i = len(p.text)
			}
			t := p.text[:i]
			p.text = p.text[i:]
			p.lastWord = t
			return t
		}
	}
}

// TestStdlibLowercase tests that all standard library package names are
// lowercase. See Issue 40065.
func TestStdlibLowercase(t *testing.T) {
	if !testenv.HasSrc() {
		t.Skipf("skipping on %s/%s, missing full GOROOT", runtime.GOOS, runtime.GOARCH)
	}

	ctxt := Default
	all, err := listStdPkgs(ctxt.GOROOT)
	if err != nil {
		t.Fatal(err)
	}

	for _, pkgname := range all {
		if strings.ToLower(pkgname) != pkgname {
			t.Errorf("package %q should not use upper-case path", pkgname)
		}
	}
}
