// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"archive/zip"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/dirhash"
	"cmd/go/internal/module"
	"cmd/go/internal/par"
)

var downloadCache par.Cache

// Download downloads the specific module version to the
// local download cache and returns the name of the directory
// corresponding to the root of the module's file tree.
func Download(mod module.Version) (dir string, err error) {
	if PkgMod == "" {
		// Do not download to current directory.
		return "", fmt.Errorf("missing modfetch.PkgMod")
	}

	// The par.Cache here avoids duplicate work but also
	// avoids conflicts from simultaneous calls by multiple goroutines
	// for the same version.
	type cached struct {
		dir string
		err error
	}
	c := downloadCache.Do(mod, func() interface{} {
		dir, err := DownloadDir(mod)
		if err != nil {
			return cached{"", err}
		}
		if files, _ := ioutil.ReadDir(dir); len(files) == 0 {
			zipfile, err := DownloadZip(mod)
			if err != nil {
				return cached{"", err}
			}
			modpath := mod.Path + "@" + mod.Version
			if err := Unzip(dir, zipfile, modpath, 0); err != nil {
				fmt.Fprintf(os.Stderr, "-> %s\n", err)
				return cached{"", err}
			}
		}
		checkSum(mod)
		return cached{dir, nil}
	}).(cached)
	return c.dir, c.err
}

var downloadZipCache par.Cache

// DownloadZip downloads the specific module version to the
// local zip cache and returns the name of the zip file.
func DownloadZip(mod module.Version) (zipfile string, err error) {
	// The par.Cache here avoids duplicate work but also
	// avoids conflicts from simultaneous calls by multiple goroutines
	// for the same version.
	type cached struct {
		zipfile string
		err     error
	}
	c := downloadZipCache.Do(mod, func() interface{} {
		zipfile, err := CachePath(mod, "zip")
		if err != nil {
			return cached{"", err}
		}
		if _, err := os.Stat(zipfile); err == nil {
			// Use it.
			// This should only happen if the mod/cache directory is preinitialized
			// or if pkg/mod/path was removed but not pkg/mod/cache/download.
			if cfg.CmdName != "mod download" {
				fmt.Fprintf(os.Stderr, "go: extracting %s %s\n", mod.Path, mod.Version)
			}
		} else {
			if err := os.MkdirAll(filepath.Dir(zipfile), 0777); err != nil {
				return cached{"", err}
			}
			if cfg.CmdName != "mod download" {
				fmt.Fprintf(os.Stderr, "go: downloading %s %s\n", mod.Path, mod.Version)
			}
			if err := downloadZip(mod, zipfile); err != nil {
				return cached{"", err}
			}
		}
		return cached{zipfile, nil}
	}).(cached)
	return c.zipfile, c.err
}

func downloadZip(mod module.Version, target string) error {
	repo, err := Lookup(mod.Path)
	if err != nil {
		return err
	}
	tmpfile, err := repo.Zip(mod.Version, os.TempDir())
	if err != nil {
		return err
	}
	defer os.Remove(tmpfile)

	// Double-check zip file looks OK.
	z, err := zip.OpenReader(tmpfile)
	if err != nil {
		return err
	}
	prefix := mod.Path + "@" + mod.Version
	for _, f := range z.File {
		if !strings.HasPrefix(f.Name, prefix) {
			z.Close()
			return fmt.Errorf("zip for %s has unexpected file %s", prefix[:len(prefix)-1], f.Name)
		}
	}
	z.Close()

	hash, err := dirhash.HashZip(tmpfile, dirhash.DefaultHash)
	if err != nil {
		return err
	}
	checkOneSum(mod, hash) // check before installing the zip file
	r, err := os.Open(tmpfile)
	if err != nil {
		return err
	}
	defer r.Close()
	w, err := os.Create(target)
	if err != nil {
		return err
	}
	if _, err := io.Copy(w, r); err != nil {
		w.Close()
		return fmt.Errorf("copying: %v", err)
	}
	if err := w.Close(); err != nil {
		return err
	}
	return ioutil.WriteFile(target+"hash", []byte(hash), 0666)
}

var GoSumFile string // path to go.sum; set by package modload

var goSum struct {
	mu        sync.Mutex
	m         map[module.Version][]string // content of go.sum file (+ go.modverify if present)
	enabled   bool                        // whether to use go.sum at all
	modverify string                      // path to go.modverify, to be deleted
}

// initGoSum initializes the go.sum data.
// It reports whether use of go.sum is now enabled.
// The goSum lock must be held.
func initGoSum() bool {
	if GoSumFile == "" {
		return false
	}
	if goSum.m != nil {
		return true
	}

	goSum.m = make(map[module.Version][]string)
	data, err := ioutil.ReadFile(GoSumFile)
	if err != nil && !os.IsNotExist(err) {
		base.Fatalf("go: %v", err)
	}
	goSum.enabled = true
	readGoSum(GoSumFile, data)

	// Add old go.modverify file.
	// We'll delete go.modverify in WriteGoSum.
	alt := strings.TrimSuffix(GoSumFile, ".sum") + ".modverify"
	if data, err := ioutil.ReadFile(alt); err == nil {
		readGoSum(alt, data)
		goSum.modverify = alt
	}
	return true
}

// emptyGoModHash is the hash of a 1-file tree containing a 0-length go.mod.
// A bug caused us to write these into go.sum files for non-modules.
// We detect and remove them.
const emptyGoModHash = "h1:G7mAYYxgmS0lVkHyy2hEOLQCFB0DlQFTMLWggykrydY="

// readGoSum parses data, which is the content of file,
// and adds it to goSum.m. The goSum lock must be held.
func readGoSum(file string, data []byte) {
	lineno := 0
	for len(data) > 0 {
		var line []byte
		lineno++
		i := bytes.IndexByte(data, '\n')
		if i < 0 {
			line, data = data, nil
		} else {
			line, data = data[:i], data[i+1:]
		}
		f := strings.Fields(string(line))
		if len(f) == 0 {
			// blank line; skip it
			continue
		}
		if len(f) != 3 {
			base.Fatalf("go: malformed go.sum:\n%s:%d: wrong number of fields %v", file, lineno, len(f))
		}
		if f[2] == emptyGoModHash {
			// Old bug; drop it.
			continue
		}
		mod := module.Version{Path: f[0], Version: f[1]}
		goSum.m[mod] = append(goSum.m[mod], f[2])
	}
}

// checkSum checks the given module's checksum.
func checkSum(mod module.Version) {
	if PkgMod == "" {
		// Do not use current directory.
		return
	}

	// Do the file I/O before acquiring the go.sum lock.
	ziphash, err := CachePath(mod, "ziphash")
	if err != nil {
		base.Fatalf("go: verifying %s@%s: %v", mod.Path, mod.Version, err)
	}
	data, err := ioutil.ReadFile(ziphash)
	if err != nil {
		if os.IsNotExist(err) {
			// This can happen if someone does rm -rf GOPATH/src/cache/download. So it goes.
			return
		}
		base.Fatalf("go: verifying %s@%s: %v", mod.Path, mod.Version, err)
	}
	h := strings.TrimSpace(string(data))
	if !strings.HasPrefix(h, "h1:") {
		base.Fatalf("go: verifying %s@%s: unexpected ziphash: %q", mod.Path, mod.Version, h)
	}

	checkOneSum(mod, h)
}

// goModSum returns the checksum for the go.mod contents.
func goModSum(data []byte) (string, error) {
	return dirhash.Hash1([]string{"go.mod"}, func(string) (io.ReadCloser, error) {
		return ioutil.NopCloser(bytes.NewReader(data)), nil
	})
}

// checkGoMod checks the given module's go.mod checksum;
// data is the go.mod content.
func checkGoMod(path, version string, data []byte) {
	h, err := goModSum(data)
	if err != nil {
		base.Fatalf("go: verifying %s %s go.mod: %v", path, version, err)
	}

	checkOneSum(module.Version{Path: path, Version: version + "/go.mod"}, h)
}

// checkOneSum checks that the recorded hash for mod is h.
func checkOneSum(mod module.Version, h string) {
	goSum.mu.Lock()
	defer goSum.mu.Unlock()
	if !initGoSum() {
		return
	}

	for _, vh := range goSum.m[mod] {
		if h == vh {
			return
		}
		if strings.HasPrefix(vh, "h1:") {
			base.Fatalf("go: verifying %s@%s: checksum mismatch\n\tdownloaded: %v\n\tgo.sum:     %v", mod.Path, mod.Version, h, vh)
		}
	}
	if len(goSum.m[mod]) > 0 {
		fmt.Fprintf(os.Stderr, "warning: verifying %s@%s: unknown hashes in go.sum: %v; adding %v", mod.Path, mod.Version, strings.Join(goSum.m[mod], ", "), h)
	}
	goSum.m[mod] = append(goSum.m[mod], h)
}

// Sum returns the checksum for the downloaded copy of the given module,
// if present in the download cache.
func Sum(mod module.Version) string {
	if PkgMod == "" {
		// Do not use current directory.
		return ""
	}

	ziphash, err := CachePath(mod, "ziphash")
	if err != nil {
		return ""
	}
	data, err := ioutil.ReadFile(ziphash)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

// WriteGoSum writes the go.sum file if it needs to be updated.
func WriteGoSum() {
	goSum.mu.Lock()
	defer goSum.mu.Unlock()
	if !initGoSum() {
		return
	}

	var mods []module.Version
	for m := range goSum.m {
		mods = append(mods, m)
	}
	module.Sort(mods)
	var buf bytes.Buffer
	for _, m := range mods {
		list := goSum.m[m]
		sort.Strings(list)
		for _, h := range list {
			fmt.Fprintf(&buf, "%s %s %s\n", m.Path, m.Version, h)
		}
	}

	data, _ := ioutil.ReadFile(GoSumFile)
	if !bytes.Equal(data, buf.Bytes()) {
		if err := ioutil.WriteFile(GoSumFile, buf.Bytes(), 0666); err != nil {
			base.Fatalf("go: writing go.sum: %v", err)
		}
	}

	if goSum.modverify != "" {
		os.Remove(goSum.modverify)
	}
}

// TrimGoSum trims go.sum to contain only the modules for which keep[m] is true.
func TrimGoSum(keep map[module.Version]bool) {
	goSum.mu.Lock()
	defer goSum.mu.Unlock()
	if !initGoSum() {
		return
	}

	for m := range goSum.m {
		// If we're keeping x@v we also keep x@v/go.mod.
		// Map x@v/go.mod back to x@v for the keep lookup.
		noGoMod := module.Version{Path: m.Path, Version: strings.TrimSuffix(m.Version, "/go.mod")}
		if !keep[m] && !keep[noGoMod] {
			delete(goSum.m, m)
		}
	}
}
