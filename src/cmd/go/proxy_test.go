// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"cmd/go/internal/modfetch"
	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/module"
	"cmd/go/internal/par"
	"cmd/go/internal/semver"
	"cmd/go/internal/txtar"
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
		fmt.Fprintf(os.Stderr, "go test proxy starting\n")
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
	})
}

var modList []module.Version

func readModList() {
	infos, err := ioutil.ReadDir("testdata/mod")
	if err != nil {
		log.Fatal(err)
	}
	for _, info := range infos {
		name := info.Name()
		if !strings.HasSuffix(name, ".txt") {
			continue
		}
		name = strings.TrimSuffix(name, ".txt")
		i := strings.LastIndex(name, "_v")
		if i < 0 {
			continue
		}
		encPath := strings.Replace(name[:i], "_", "/", -1)
		path, err := module.DecodePath(encPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "go proxy_test: %v\n", err)
			continue
		}
		encVers := name[i+1:]
		vers, err := module.DecodeVersion(encVers)
		if err != nil {
			fmt.Fprintf(os.Stderr, "go proxy_test: %v\n", err)
			continue
		}
		modList = append(modList, module.Version{Path: path, Version: vers})
	}
}

var zipCache par.Cache

// proxyHandler serves the Go module proxy protocol.
// See the proxy section of https://research.swtch.com/vgo-module.
func proxyHandler(w http.ResponseWriter, r *http.Request) {
	if !strings.HasPrefix(r.URL.Path, "/mod/") {
		http.NotFound(w, r)
		return
	}
	path := strings.TrimPrefix(r.URL.Path, "/mod/")
	i := strings.Index(path, "/@v/")
	if i < 0 {
		http.NotFound(w, r)
		return
	}
	enc, file := path[:i], path[i+len("/@v/"):]
	path, err := module.DecodePath(enc)
	if err != nil {
		fmt.Fprintf(os.Stderr, "go proxy_test: %v\n", err)
		http.NotFound(w, r)
		return
	}
	if file == "list" {
		n := 0
		for _, m := range modList {
			if m.Path == path && !modfetch.IsPseudoVersion(m.Version) {
				if err := module.Check(m.Path, m.Version); err == nil {
					fmt.Fprintf(w, "%s\n", m.Version)
					n++
				}
			}
		}
		if n == 0 {
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
	vers, err := module.DecodeVersion(encVers)
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
				if modfetch.IsPseudoVersion(m.Version) {
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

	a := readArchive(path, vers)
	if a == nil {
		fmt.Fprintf(os.Stderr, "go proxy: no archive %s %s\n", path, vers)
		http.Error(w, "cannot load archive", 500)
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
				if strings.HasPrefix(f.Name, ".") {
					continue
				}
				zf, err := z.Create(path + "@" + vers + "/" + f.Name)
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
			fmt.Fprintf(os.Stderr, "go proxy: %v\n", c.err)
			http.Error(w, c.err.Error(), 500)
			return
		}
		w.Write(c.zip)
		return

	}
	http.NotFound(w, r)
}

func findHash(m module.Version) string {
	a := readArchive(m.Path, m.Version)
	if a == nil {
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

func readArchive(path, vers string) *txtar.Archive {
	enc, err := module.EncodePath(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "go proxy: %v\n", err)
		return nil
	}
	encVers, err := module.EncodeVersion(vers)
	if err != nil {
		fmt.Fprintf(os.Stderr, "go proxy: %v\n", err)
		return nil
	}

	prefix := strings.Replace(enc, "/", "_", -1)
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
	return a
}
