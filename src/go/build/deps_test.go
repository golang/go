// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file exercises the import parser but also checks that
// some low-level packages do not have new dependencies added.

package build

import (
	"bytes"
	"fmt"
	"go/token"
	"internal/dag"
	"internal/testenv"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
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
// "a < b" means package b can import package a.
//
// See `go doc internal/dag` for the full syntax.
//
// All-caps names are pseudo-names for specific points
// in the dependency lattice.
var depsRules = `
	# No dependencies allowed for any of these packages.
	NONE
	< unsafe
	< cmp,
	  container/list,
	  container/ring,
	  internal/byteorder,
	  internal/cfg,
	  internal/coverage,
	  internal/coverage/rtcov,
	  internal/coverage/uleb128,
	  internal/coverage/calloc,
	  internal/cpu,
	  internal/goarch,
	  internal/godebugs,
	  internal/goexperiment,
	  internal/goos,
	  internal/goversion,
	  internal/nettrace,
	  internal/platform,
	  internal/profilerecord,
	  internal/syslist,
	  internal/trace/traceviewer/format,
	  log/internal,
	  math/bits,
	  structs,
	  unicode,
	  unicode/utf8,
	  unicode/utf16;

	internal/goarch < internal/abi;
	internal/byteorder, internal/goarch < internal/chacha8rand;

	# RUNTIME is the core runtime group of packages, all of them very light-weight.
	internal/abi,
	internal/chacha8rand,
	internal/coverage/rtcov,
	internal/cpu,
	internal/goarch,
	internal/godebugs,
	internal/goexperiment,
	internal/goos,
	internal/profilerecord,
	math/bits,
	structs
	< internal/bytealg
	< internal/stringslite
	< internal/itoa
	< internal/unsafeheader
	< internal/race
	< internal/msan
	< internal/asan
	< internal/runtime/sys
	< internal/runtime/syscall
	< internal/runtime/atomic
	< internal/runtime/exithook
	< internal/runtime/math
	< internal/runtime/maps
	< runtime
	< sync/atomic
	< internal/weak
	< sync
	< internal/bisect
	< internal/godebug
	< internal/reflectlite
	< errors
	< internal/oserror;

	cmp, runtime, math/bits
	< iter
	< maps, slices;

	internal/oserror, maps, slices
	< RUNTIME;

	RUNTIME
	< sort
	< container/heap;

	RUNTIME
	< io;

	RUNTIME
	< arena;

	syscall !< io;
	reflect !< sort;

	RUNTIME, unicode/utf8
	< path;

	unicode !< path;

	# SYSCALL is RUNTIME plus the packages necessary for basic system calls.
	RUNTIME, unicode/utf8, unicode/utf16
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

	TIME, io, path, slices
	< io/fs;

	# MATH is RUNTIME plus the basic math packages.
	RUNTIME
	< math
	< MATH;

	unicode !< math;

	MATH
	< math/cmplx;

	MATH
	< math/rand, math/rand/v2;

	MATH
	< runtime/metrics;

	RUNTIME, math/rand/v2
	< internal/concurrent;

	MATH, unicode/utf8
	< strconv;

	unicode !< strconv;

	# STR is basic string and buffer manipulation.
	RUNTIME, io, unicode/utf8, unicode/utf16, unicode
	< bytes, strings
	< bufio;

	bufio, path, strconv
	< STR;

	RUNTIME, internal/concurrent
	< unique;

	# OS is basic OS access, including helpers (path/filepath, os/exec, etc).
	# OS includes string routines, but those must be layered above package os.
	# OS does not include reflection.
	io/fs
	< internal/testlog
	< internal/poll
	< internal/filepathlite
	< os
	< os/signal;

	io/fs
	< embed;

	unicode, fmt !< net, os, os/signal;

	os/signal, internal/filepathlite, STR
	< path/filepath
	< io/ioutil;

	path/filepath, internal/godebug < os/exec;

	io/ioutil, os/exec, os/signal
	< OS;

	reflect !< OS;

	OS
	< golang.org/x/sys/cpu;

	# FMT is OS (which includes string routines) plus reflect and fmt.
	# It does not include package log, which should be avoided in core packages.
	arena, strconv, unicode
	< reflect;

	os, reflect
	< internal/fmtsort
	< fmt;

	OS, fmt
	< FMT;

	log !< FMT;

	# Misc packages needing only FMT.
	FMT
	< html,
	  internal/dag,
	  internal/goroot,
	  internal/types/errors,
	  mime/quotedprintable,
	  net/internal/socktest,
	  net/url,
	  runtime/trace,
	  text/scanner,
	  text/tabwriter;

	io, reflect
	< internal/saferio;

	# encodings
	# core ones do not use fmt.
	io, strconv, slices
	< encoding;

	encoding, reflect
	< encoding/binary
	< encoding/base32, encoding/base64;

	FMT, encoding < flag;

	fmt !< encoding/base32, encoding/base64;

	FMT, encoding/base32, encoding/base64, internal/saferio
	< encoding/ascii85, encoding/csv, encoding/gob, encoding/hex,
	  encoding/json, encoding/pem, encoding/xml, mime;

	# hashes
	io
	< hash
	< hash/adler32, hash/crc32, hash/crc64, hash/fnv;

	# math/big
	FMT, math/rand
	< math/big;

	# compression
	FMT, encoding/binary, hash/adler32, hash/crc32, sort
	< compress/bzip2, compress/flate, compress/lzw, internal/zstd
	< archive/zip, compress/gzip, compress/zlib;

	# templates
	FMT
	< text/template/parse;

	net/url, text/template/parse
	< text/template
	< internal/lazytemplate;

	# regexp
	FMT, sort
	< regexp/syntax
	< regexp
	< internal/lazyregexp;

	encoding/json, html, text/template, regexp
	< html/template;

	# suffix array
	encoding/binary, regexp
	< index/suffixarray;

	# executable parsing
	FMT, encoding/binary, compress/zlib, internal/saferio, internal/zstd, sort
	< runtime/debug
	< debug/dwarf
	< debug/elf, debug/gosym, debug/macho, debug/pe, debug/plan9obj, internal/xcoff
	< debug/buildinfo
	< DEBUG;

	# go parser and friends.
	FMT, sort
	< internal/gover
	< go/version
	< go/token
	< go/scanner
	< go/ast
	< go/internal/typeparams;

	FMT
	< go/build/constraint;

	FMT, sort
	< go/doc/comment;

	go/internal/typeparams, go/build/constraint
	< go/parser;

	go/doc/comment, go/parser, text/tabwriter
	< go/printer
	< go/format;

	math/big, go/token
	< go/constant;

	FMT, internal/goexperiment
	< internal/buildcfg;

	container/heap, go/constant, go/parser, internal/buildcfg, internal/goversion, internal/types/errors
	< go/types;

	# The vast majority of standard library packages should not be resorting to regexp.
	# go/types is a good chokepoint. It shouldn't use regexp, nor should anything
	# that is low-enough level to be used by go/types.
	regexp !< go/types;

	go/doc/comment, go/parser, internal/lazyregexp, text/template
	< go/doc;

	go/build/constraint, go/doc, go/parser, internal/buildcfg, internal/goroot, internal/goversion, internal/platform, internal/syslist
	< go/build;

	# databases
	FMT
	< database/sql/internal
	< database/sql/driver;

	database/sql/driver, math/rand/v2 < database/sql;

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
	< runtime/msan, runtime/asan;

	# runtime/race
	NONE < runtime/race/internal/amd64v1;
	NONE < runtime/race/internal/amd64v3;
	CGO, runtime/race/internal/amd64v1, runtime/race/internal/amd64v3 < runtime/race;

	# Bulk of the standard library must not use cgo.
	# The prohibition stops at net and os/user.
	C !< fmt, go/types, CRYPTO-MATH, log/slog;

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

	internal/bytealg, internal/itoa, math/bits, slices, strconv, unique
	< net/netip;

	# net is unavoidable when doing any networking,
	# so large dependencies must be kept out.
	# This is a long-looking list but most of these
	# are small with few dependencies.
	CGO,
	golang.org/x/net/dns/dnsmessage,
	golang.org/x/net/lif,
	golang.org/x/net/route,
	internal/godebug,
	internal/nettrace,
	internal/poll,
	internal/singleflight,
	net/netip,
	os,
	sort
	< net;

	fmt, unicode !< net;
	math/rand !< net; # net uses runtime instead

	# NET is net plus net-helper packages.
	FMT, net
	< net/textproto;

	mime, net/textproto, net/url
	< NET;

	# logging - most packages should not import; http and up is allowed
	FMT, log/internal
	< log;

	log, log/slog !< crypto/tls, database/sql, go/importer, testing;

	FMT, log, net
	< log/syslog;

	RUNTIME
	< log/slog/internal, log/slog/internal/buffer;

	FMT,
	encoding, encoding/json,
	log, log/internal,
	log/slog/internal, log/slog/internal/buffer,
	slices
	< log/slog
	< log/slog/internal/slogtest, log/slog/internal/benchmarks;

	NET, log
	< net/mail;

	NONE < crypto/internal/impl;

	# FIPS is the FIPS 140 module.
	# It must not depend on external crypto packages.
	# Internal packages imported by FIPS might need to retain
	# backwards compatibility with older versions of the module.
	STR, crypto/internal/impl
	< crypto/internal/fips
	< crypto/internal/fips/subtle
	< crypto/internal/fips/sha256
	< crypto/internal/fips/sha512
	< crypto/internal/fips/sha3
	< crypto/internal/fips/hmac
	< crypto/internal/fips/check
	< FIPS;

	FIPS < crypto/internal/fips/check/checktest;

	NONE < crypto/internal/boring/sig, crypto/internal/boring/syso;
	sync/atomic < crypto/internal/boring/bcache, crypto/internal/boring/fipstls;
	crypto/internal/boring/sig, crypto/internal/boring/fipstls < crypto/tls/fipsonly;

	# CRYPTO is core crypto algorithms - no cgo, fmt, net.
	FIPS,
	crypto/internal/boring/sig,
	crypto/internal/boring/syso,
	golang.org/x/sys/cpu,
	hash, embed
	< crypto
	< crypto/subtle
	< crypto/internal/alias
	< crypto/cipher;

	crypto/cipher,
	crypto/internal/boring/bcache
	< crypto/internal/boring
	< crypto/boring;

	crypto/internal/alias, math/rand/v2
	< crypto/internal/randutil
	< crypto/internal/nistec/fiat
	< crypto/internal/nistec
	< crypto/internal/edwards25519/field
	< crypto/internal/edwards25519;

	crypto/boring
	< crypto/aes, crypto/des, crypto/hmac, crypto/md5, crypto/rc4,
	  crypto/sha1, crypto/sha256, crypto/sha512;

	crypto/boring, crypto/internal/edwards25519/field
	< crypto/ecdh;

	# Unfortunately, stuck with reflect via encoding/binary.
	encoding/binary, crypto/boring < golang.org/x/crypto/sha3;

	crypto/aes,
	crypto/des,
	crypto/ecdh,
	crypto/hmac,
	crypto/internal/edwards25519,
	crypto/md5,
	crypto/rc4,
	crypto/sha1,
	crypto/sha256,
	crypto/sha512,
	golang.org/x/crypto/sha3
	< CRYPTO;

	CGO, fmt, net !< CRYPTO;

	# CRYPTO-MATH is core bignum-based crypto - no cgo, net; fmt now ok.
	CRYPTO, FMT, math/big
	< crypto/internal/boring/bbig
	< crypto/rand
	< crypto/internal/mlkem768
	< crypto/ed25519
	< encoding/asn1
	< golang.org/x/crypto/cryptobyte/asn1
	< golang.org/x/crypto/cryptobyte
	< crypto/internal/bigmod
	< crypto/dsa, crypto/elliptic, crypto/rsa
	< crypto/ecdsa
	< CRYPTO-MATH;

	CGO, net !< CRYPTO-MATH;

	# TLS, Prince of Dependencies.
	CRYPTO-MATH, NET, container/list, encoding/hex, encoding/pem
	< golang.org/x/crypto/internal/alias
	< golang.org/x/crypto/internal/subtle
	< golang.org/x/crypto/chacha20
	< golang.org/x/crypto/internal/poly1305
	< golang.org/x/crypto/chacha20poly1305
	< golang.org/x/crypto/hkdf
	< crypto/internal/hpke
	< crypto/x509/internal/macos
	< crypto/x509/pkix;

	crypto/internal/boring/fipstls, crypto/x509/pkix
	< crypto/x509
	< crypto/tls;

	# crypto-aware packages

	DEBUG, go/build, go/types, text/scanner, crypto/md5
	< internal/pkgbits, internal/exportdata
	< go/internal/gcimporter, go/internal/gccgoimporter, go/internal/srcimporter
	< go/importer;

	NET, crypto/rand, mime/quotedprintable
	< mime/multipart;

	crypto/tls
	< net/smtp;

	crypto/rand
	< hash/maphash; # for purego implementation

	# HTTP, King of Dependencies.

	FMT
	< golang.org/x/net/http2/hpack
	< net/http/internal, net/http/internal/ascii, net/http/internal/testcert;

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
	net/http/internal/ascii,
	net/http/internal/testcert,
	net/http/httptrace,
	mime/multipart,
	log
	< net/http;

	# HTTP-aware packages

	encoding/json, net/http
	< expvar;

	net/http, net/http/internal/ascii
	< net/http/cookiejar, net/http/httputil;

	net/http, flag
	< net/http/httptest;

	net/http, regexp
	< net/http/cgi
	< net/http/fcgi;

	# Profiling
	FMT, compress/gzip, encoding/binary, sort, text/tabwriter
	< runtime/pprof;

	OS, compress/gzip, internal/lazyregexp
	< internal/profile;

	html, internal/profile, net/http, runtime/pprof, runtime/trace
	< net/http/pprof;

	# RPC
	encoding/gob, encoding/json, go/token, html/template, net/http
	< net/rpc
	< net/rpc/jsonrpc;

	# System Information
	bufio, bytes, internal/cpu, io, os, strings, sync
	< internal/sysinfo;

	# Test-only
	log
	< testing/iotest
	< testing/fstest;

	FMT, flag, math/rand
	< testing/quick;

	FMT, DEBUG, flag, runtime/trace, internal/sysinfo, math/rand
	< testing;

	log/slog, testing
	< testing/slogtest;

	FMT, crypto/sha256, encoding/json, go/ast, go/parser, go/token,
	internal/godebug, math/rand, encoding/hex, crypto/sha256
	< internal/fuzz;

	OS, flag, testing, internal/cfg, internal/platform, internal/goroot
	< internal/testenv;

	OS, encoding/base64
	< internal/obscuretestdata;

	CGO, OS, fmt
	< internal/testpty;

	NET, testing, math/rand
	< golang.org/x/net/nettest;

	syscall
	< os/exec/internal/fdtest;

	FMT, sort
	< internal/diff;

	FMT
	< internal/txtar;

	CRYPTO-MATH, testing, internal/testenv
	< crypto/internal/cryptotest;

	CGO, FMT
	< crypto/rand/internal/seccomp;

	# v2 execution trace parser.
	FMT
	< internal/trace/event;

	internal/trace/event
	< internal/trace/event/go122;

	FMT, io, internal/trace/event/go122
	< internal/trace/version;

	FMT, encoding/binary, internal/trace/version
	< internal/trace/raw;

	FMT, internal/trace/event, internal/trace/version, io, sort, encoding/binary
	< internal/trace/internal/oldtrace;

	FMT, encoding/binary, internal/trace/version, internal/trace/internal/oldtrace, container/heap, math/rand
	< internal/trace;

	regexp, internal/trace, internal/trace/raw, internal/txtar
	< internal/trace/testtrace;

	regexp, internal/txtar, internal/trace, internal/trace/raw
	< internal/trace/internal/testgen/go122;

	# cmd/trace dependencies.
	FMT,
	embed,
	encoding/json,
	html/template,
	internal/profile,
	internal/trace,
	internal/trace/traceviewer/format,
	net/http
	< internal/trace/traceviewer;

	# Coverage.
	FMT, hash/fnv, encoding/binary, regexp, sort, text/tabwriter,
	internal/coverage, internal/coverage/uleb128
	< internal/coverage/cmerge,
	  internal/coverage/pods,
	  internal/coverage/slicereader,
	  internal/coverage/slicewriter;

	internal/coverage/slicereader, internal/coverage/slicewriter
	< internal/coverage/stringtab
	< internal/coverage/decodecounter, internal/coverage/decodemeta,
	  internal/coverage/encodecounter, internal/coverage/encodemeta;

	internal/coverage/cmerge
	< internal/coverage/cformat;

	internal/coverage, crypto/sha256, FMT
	< cmd/internal/cov/covcmd;

	encoding/json,
	runtime/debug,
	internal/coverage/calloc,
	internal/coverage/cformat,
	internal/coverage/decodecounter, internal/coverage/decodemeta,
	internal/coverage/encodecounter, internal/coverage/encodemeta,
	internal/coverage/pods
	< internal/coverage/cfile
	< runtime/coverage;

	internal/coverage/cfile, internal/fuzz, internal/testlog, runtime/pprof, regexp
	< testing/internal/testdeps;

	# Test-only packages can have anything they want
	CGO, internal/syscall/unix < net/internal/cgotest;


`

// listStdPkgs returns the same list of packages as "go list std".
func listStdPkgs(goroot string) ([]string, error) {
	// Based on cmd/go's matchPackages function.
	var pkgs []string

	src := filepath.Join(goroot, "src") + string(filepath.Separator)
	walkFn := func(path string, d fs.DirEntry, err error) error {
		if err != nil || !d.IsDir() || path == src {
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
	if err := filepath.WalkDir(src, walkFn); err != nil {
		return nil, err
	}
	return pkgs, nil
}

func TestDependencies(t *testing.T) {
	testenv.MustHaveSource(t)

	ctxt := Default
	all, err := listStdPkgs(ctxt.GOROOT)
	if err != nil {
		t.Fatal(err)
	}
	slices.Sort(all)

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
		var bad []string
		for _, imp := range imports {
			sawImport[pkg][imp] = true
			if !policy.HasEdge(pkg, imp) {
				bad = append(bad, imp)
			}
		}
		if bad != nil {
			t.Errorf("unexpected dependency: %s imports %v", pkg, bad)
		}
	}
}

var buildIgnore = []byte("\n//go:build ignore")

func findImports(pkg string) ([]string, error) {
	vpkg := pkg
	if strings.HasPrefix(pkg, "golang.org") {
		vpkg = "vendor/" + pkg
	}
	dir := filepath.Join(Default.GOROOT, "src", vpkg)
	files, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var imports []string
	var haveImport = map[string]bool{}
	if pkg == "crypto/internal/boring" {
		haveImport["C"] = true // kludge: prevent C from appearing in crypto/internal/boring imports
	}
	fset := token.NewFileSet()
	for _, file := range files {
		name := file.Name()
		if name == "slice_go14.go" || name == "slice_go18.go" {
			// These files are for compiler bootstrap with older versions of Go and not built in the standard build.
			continue
		}
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		info := fileInfo{
			name: filepath.Join(dir, name),
			fset: fset,
		}
		f, err := os.Open(info.name)
		if err != nil {
			return nil, err
		}
		err = readGoInfo(f, &info)
		f.Close()
		if err != nil {
			return nil, fmt.Errorf("reading %v: %v", name, err)
		}
		if info.parsed.Name.Name == "main" {
			continue
		}
		if bytes.Contains(info.header, buildIgnore) {
			continue
		}
		for _, imp := range info.imports {
			path := imp.path
			if !haveImport[path] {
				haveImport[path] = true
				imports = append(imports, path)
			}
		}
	}
	slices.Sort(imports)
	return imports, nil
}

// depsPolicy returns a map m such that m[p][d] == true when p can import d.
func depsPolicy(t *testing.T) *dag.Graph {
	g, err := dag.Parse(depsRules)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// TestStdlibLowercase tests that all standard library package names are
// lowercase. See Issue 40065.
func TestStdlibLowercase(t *testing.T) {
	testenv.MustHaveSource(t)

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

// TestFindImports tests that findImports works.  See #43249.
func TestFindImports(t *testing.T) {
	imports, err := findImports("go/build")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("go/build imports %q", imports)
	want := []string{"bytes", "os", "path/filepath", "strings"}
wantLoop:
	for _, w := range want {
		for _, imp := range imports {
			if imp == w {
				continue wantLoop
			}
		}
		t.Errorf("expected to find %q in import list", w)
	}
}
