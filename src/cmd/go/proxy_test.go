// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"

	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/par"

	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
	"golang.org/x/mod/sumdb"
	"golang.org/x/mod/sumdb/dirhash"
	"golang.org/x/tools/txtar"
)

var (
	proxyAddr = flag.String("proxy", "", "run proxy on this network address instead of running any tests")
	proxyURL  string
)

var proxyOnce sync.Once

// StartProxy starts the Go module proxy running on *proxyAddr (like "localhost:1234")
// and sets proxyURL to the GOPROXY setting to use to access the proxy.
// Subsequent calls are no-ops.
//
// The proxy serves from testdata/mod. See testdata/mod/README.
func StartProxy() {
	proxyOnce.Do(func() {
		readModList()
		addr := *proxyAddr
		if addr == "" {
			addr = "localhost:0"
		}
		l, err := net.Listen("tcp", addr)
		if err != nil {
			log.Fatal(err)
		}
		*proxyAddr = l.Addr().String()
		proxyURL = "http://" + *proxyAddr + "/mod"
		fmt.Fprintf(os.Stderr, "go test proxy running at GOPROXY=%s\n", proxyURL)
		go func() {
			log.Fatalf("go proxy: http.Serve: %v", http.Serve(l, http.HandlerFunc(proxyHandler)))
		}()

		// Prepopulate main sumdb.
		for _, mod := range modList {
			sumdbOps.Lookup(nil, mod)
		}
	})
}

var modList []module.Version

func readModList() {
	files, err := os.ReadDir("testdata/mod")
	if err != nil {
		log.Fatal(err)
	}
	for _, f := range files {
		name := f.Name()
		if !strings.HasSuffix(name, ".txt") {
			continue
		}
		name = strings.TrimSuffix(name, ".txt")
		i := strings.LastIndex(name, "_v")
		if i < 0 {
			continue
		}
		encPath := strings.ReplaceAll(name[:i], "_", "/")
		path, err := module.UnescapePath(encPath)
		if err != nil {
			if testing.Verbose() && encPath != "example.com/invalidpath/v1" {
				fmt.Fprintf(os.Stderr, "go proxy_test: %v\n", err)
			}
			continue
		}
		encVers := name[i+1:]
		vers, err := module.UnescapeVersion(encVers)
		if err != nil {
			fmt.Fprintf(os.Stderr, "go proxy_test: %v\n", err)
			continue
		}
		modList = append(modList, module.Version{Path: path, Version: vers})
	}
}

var zipCache par.Cache

const (
	testSumDBName        = "localhost.localdev/sumdb"
	testSumDBVerifierKey = "localhost.localdev/sumdb+00000c67+AcTrnkbUA+TU4heY3hkjiSES/DSQniBqIeQ/YppAUtK6"
	testSumDBSignerKey   = "PRIVATE+KEY+localhost.localdev/sumdb+00000c67+AXu6+oaVaOYuQOFrf1V59JK1owcFlJcHwwXHDfDGxSPk"
)

var (
	sumdbOps    = sumdb.NewTestServer(testSumDBSignerKey, proxyGoSum)
	sumdbServer = sumdb.NewServer(sumdbOps)

	sumdbWrongOps    = sumdb.NewTestServer(testSumDBSignerKey, proxyGoSumWrong)
	sumdbWrongServer = sumdb.NewServer(sumdbWrongOps)
)

// proxyHandler serves the Go module proxy protocol.
// See the proxy section of https://research.swtch.com/vgo-module.
func proxyHandler(w http.ResponseWriter, r *http.Request) {
	if !strings.HasPrefix(r.URL.Path, "/mod/") {
		http.NotFound(w, r)
		return
	}
	path := r.URL.Path[len("/mod/"):]

	// /mod/invalid returns faulty responses.
	if strings.HasPrefix(path, "invalid/") {
		w.Write([]byte("invalid"))
		return
	}

	// Next element may opt into special behavior.
	if j := strings.Index(path, "/"); j >= 0 {
		n, err := strconv.Atoi(path[:j])
		if err == nil && n >= 200 {
			w.WriteHeader(n)
			return
		}
		if strings.HasPrefix(path, "sumdb-") {
			n, err := strconv.Atoi(path[len("sumdb-"):j])
			if err == nil && n >= 200 {
				if strings.HasPrefix(path[j:], "/sumdb/") {
					w.WriteHeader(n)
					return
				}
				path = path[j+1:]
			}
		}
	}

	// Request for $GOPROXY/sumdb-direct is direct sumdb access.
	// (Client thinks it is talking directly to a sumdb.)
	if strings.HasPrefix(path, "sumdb-direct/") {
		r.URL.Path = path[len("sumdb-direct"):]
		sumdbServer.ServeHTTP(w, r)
		return
	}

	// Request for $GOPROXY/sumdb-wrong is direct sumdb access
	// but all the hashes are wrong.
	// (Client thinks it is talking directly to a sumdb.)
	if strings.HasPrefix(path, "sumdb-wrong/") {
		r.URL.Path = path[len("sumdb-wrong"):]
		sumdbWrongServer.ServeHTTP(w, r)
		return
	}

	// Request for $GOPROXY/redirect/<count>/... goes to redirects.
	if strings.HasPrefix(path, "redirect/") {
		path = path[len("redirect/"):]
		if j := strings.Index(path, "/"); j >= 0 {
			count, err := strconv.Atoi(path[:j])
			if err != nil {
				return
			}

			// The last redirect.
			if count <= 1 {
				http.Redirect(w, r, fmt.Sprintf("/mod/%s", path[j+1:]), 302)
				return
			}
			http.Redirect(w, r, fmt.Sprintf("/mod/redirect/%d/%s", count-1, path[j+1:]), 302)
			return
		}
	}

	// Request for $GOPROXY/sumdb/<name>/supported
	// is checking whether it's OK to access sumdb via the proxy.
	if path == "sumdb/"+testSumDBName+"/supported" {
		w.WriteHeader(200)
		return
	}

	// Request for $GOPROXY/sumdb/<name>/... goes to sumdb.
	if sumdbPrefix := "sumdb/" + testSumDBName + "/"; strings.HasPrefix(path, sumdbPrefix) {
		r.URL.Path = path[len(sumdbPrefix)-1:]
		sumdbServer.ServeHTTP(w, r)
		return
	}

	// Module proxy request: /mod/path/@latest
	// Rewrite to /mod/path/@v/<latest>.info where <latest> is the semantically
	// latest version, including pseudo-versions.
	if i := strings.LastIndex(path, "/@latest"); i >= 0 {
		enc := path[:i]
		modPath, err := module.UnescapePath(enc)
		if err != nil {
			if testing.Verbose() {
				fmt.Fprintf(os.Stderr, "go proxy_test: %v\n", err)
			}
			http.NotFound(w, r)
			return
		}

		// Imitate what "latest" does in direct mode and what proxy.golang.org does.
		// Use the latest released version.
		// If there is no released version, use the latest prereleased version.
		// Otherwise, use the latest pseudoversion.
		var latestRelease, latestPrerelease, latestPseudo string
		for _, m := range modList {
			if m.Path != modPath {
				continue
			}
			if module.IsPseudoVersion(m.Version) && (latestPseudo == "" || semver.Compare(latestPseudo, m.Version) > 0) {
				latestPseudo = m.Version
			} else if semver.Prerelease(m.Version) != "" && (latestPrerelease == "" || semver.Compare(latestPrerelease, m.Version) > 0) {
				latestPrerelease = m.Version
			} else if latestRelease == "" || semver.Compare(latestRelease, m.Version) > 0 {
				latestRelease = m.Version
			}
		}
		var latest string
		if latestRelease != "" {
			latest = latestRelease
		} else if latestPrerelease != "" {
			latest = latestPrerelease
		} else if latestPseudo != "" {
			latest = latestPseudo
		} else {
			http.NotFound(w, r)
			return
		}

		encVers, err := module.EscapeVersion(latest)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		path = fmt.Sprintf("%s/@v/%s.info", enc, encVers)
	}

	// Module proxy request: /mod/path/@v/version[.suffix]
	i := strings.Index(path, "/@v/")
	if i < 0 {
		http.NotFound(w, r)
		return
	}
	enc, file := path[:i], path[i+len("/@v/"):]
	path, err := module.UnescapePath(enc)
	if err != nil {
		if testing.Verbose() {
			fmt.Fprintf(os.Stderr, "go proxy_test: %v\n", err)
		}
		http.NotFound(w, r)
		return
	}
	if file == "list" {
		// list returns a list of versions, not including pseudo-versions.
		// If the module has no tagged versions, we should serve an empty 200.
		// If the module doesn't exist, we should serve 404 or 410.
		found := false
		for _, m := range modList {
			if m.Path != path {
				continue
			}
			found = true
			if !module.IsPseudoVersion(m.Version) {
				if err := module.Check(m.Path, m.Version); err == nil {
					fmt.Fprintf(w, "%s\n", m.Version)
				}
			}
		}
		if !found {
			http.NotFound(w, r)
		}
		return
	}

	i = strings.LastIndex(file, ".")
	if i < 0 {
		http.NotFound(w, r)
		return
	}
	encVers, ext := file[:i], file[i+1:]
	vers, err := module.UnescapeVersion(encVers)
	if err != nil {
		fmt.Fprintf(os.Stderr, "go proxy_test: %v\n", err)
		http.NotFound(w, r)
		return
	}

	if codehost.AllHex(vers) {
		var best string
		// Convert commit hash (only) to known version.
		// Use latest version in semver priority, to match similar logic
		// in the repo-based module server (see modfetch.(*codeRepo).convert).
		for _, m := range modList {
			if m.Path == path && semver.Compare(best, m.Version) < 0 {
				var hash string
				if module.IsPseudoVersion(m.Version) {
					hash = m.Version[strings.LastIndex(m.Version, "-")+1:]
				} else {
					hash = findHash(m)
				}
				if strings.HasPrefix(hash, vers) || strings.HasPrefix(vers, hash) {
					best = m.Version
				}
			}
		}
		if best != "" {
			vers = best
		}
	}

	a, err := readArchive(path, vers)
	if err != nil {
		if testing.Verbose() {
			fmt.Fprintf(os.Stderr, "go proxy: no archive %s %s: %v\n", path, vers, err)
		}
		if errors.Is(err, fs.ErrNotExist) {
			http.NotFound(w, r)
		} else {
			http.Error(w, "cannot load archive", 500)
		}
		return
	}

	switch ext {
	case "info", "mod":
		want := "." + ext
		for _, f := range a.Files {
			if f.Name == want {
				w.Write(f.Data)
				return
			}
		}

	case "zip":
		type cached struct {
			zip []byte
			err error
		}
		c := zipCache.Do(a, func() interface{} {
			var buf bytes.Buffer
			z := zip.NewWriter(&buf)
			for _, f := range a.Files {
				if f.Name == ".info" || f.Name == ".mod" || f.Name == ".zip" {
					continue
				}
				var zipName string
				if strings.HasPrefix(f.Name, "/") {
					zipName = f.Name[1:]
				} else {
					zipName = path + "@" + vers + "/" + f.Name
				}
				zf, err := z.Create(zipName)
				if err != nil {
					return cached{nil, err}
				}
				if _, err := zf.Write(f.Data); err != nil {
					return cached{nil, err}
				}
			}
			if err := z.Close(); err != nil {
				return cached{nil, err}
			}
			return cached{buf.Bytes(), nil}
		}).(cached)

		if c.err != nil {
			if testing.Verbose() {
				fmt.Fprintf(os.Stderr, "go proxy: %v\n", c.err)
			}
			http.Error(w, c.err.Error(), 500)
			return
		}
		w.Write(c.zip)
		return

	}
	http.NotFound(w, r)
}

func findHash(m module.Version) string {
	a, err := readArchive(m.Path, m.Version)
	if err != nil {
		return ""
	}
	var data []byte
	for _, f := range a.Files {
		if f.Name == ".info" {
			data = f.Data
			break
		}
	}
	var info struct{ Short string }
	json.Unmarshal(data, &info)
	return info.Short
}

var archiveCache par.Cache

var cmdGoDir, _ = os.Getwd()

func readArchive(path, vers string) (*txtar.Archive, error) {
	enc, err := module.EscapePath(path)
	if err != nil {
		return nil, err
	}
	encVers, err := module.EscapeVersion(vers)
	if err != nil {
		return nil, err
	}

	prefix := strings.ReplaceAll(enc, "/", "_")
	name := filepath.Join(cmdGoDir, "testdata/mod", prefix+"_"+encVers+".txt")
	a := archiveCache.Do(name, func() interface{} {
		a, err := txtar.ParseFile(name)
		if err != nil {
			if testing.Verbose() || !os.IsNotExist(err) {
				fmt.Fprintf(os.Stderr, "go proxy: %v\n", err)
			}
			a = nil
		}
		return a
	}).(*txtar.Archive)
	if a == nil {
		return nil, fs.ErrNotExist
	}
	return a, nil
}

// proxyGoSum returns the two go.sum lines for path@vers.
func proxyGoSum(path, vers string) ([]byte, error) {
	a, err := readArchive(path, vers)
	if err != nil {
		return nil, err
	}
	var names []string
	files := make(map[string][]byte)
	var gomod []byte
	for _, f := range a.Files {
		if strings.HasPrefix(f.Name, ".") {
			if f.Name == ".mod" {
				gomod = f.Data
			}
			continue
		}
		name := path + "@" + vers + "/" + f.Name
		names = append(names, name)
		files[name] = f.Data
	}
	h1, err := dirhash.Hash1(names, func(name string) (io.ReadCloser, error) {
		data := files[name]
		return io.NopCloser(bytes.NewReader(data)), nil
	})
	if err != nil {
		return nil, err
	}
	h1mod, err := dirhash.Hash1([]string{"go.mod"}, func(string) (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(gomod)), nil
	})
	if err != nil {
		return nil, err
	}
	data := []byte(fmt.Sprintf("%s %s %s\n%s %s/go.mod %s\n", path, vers, h1, path, vers, h1mod))
	return data, nil
}

// proxyGoSumWrong returns the wrong lines.
func proxyGoSumWrong(path, vers string) ([]byte, error) {
	data := []byte(fmt.Sprintf("%s %s %s\n%s %s/go.mod %s\n", path, vers, "h1:wrong", path, vers, "h1:wrong"))
	return data, nil
}
