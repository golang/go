// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vgo

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

	"cmd/go/internal/base"
	"cmd/go/internal/dirhash"
	"cmd/go/internal/modfetch"
	"cmd/go/internal/module"
	"cmd/go/internal/semver"
)

// fetch returns the directory in the local download cache
// holding the root of mod's source tree.
// It downloads the module if needed.
func fetch(mod module.Version) (dir string, err error) {
	if r := replaced(mod); r != nil {
		if r.New.Version == "" {
			dir = r.New.Path
			if !filepath.IsAbs(dir) {
				dir = filepath.Join(ModRoot, dir)
			}
			return dir, nil
		}
		mod = r.New
	}

	modpath := mod.Path + "@" + mod.Version
	dir = filepath.Join(srcV, modpath)
	if files, _ := ioutil.ReadDir(dir); len(files) == 0 {
		zipfile := filepath.Join(srcV, "cache", mod.Path, "@v", mod.Version+".zip")
		if _, err := os.Stat(zipfile); err == nil {
			// Use it.
			// This should only happen if the v/cache directory is preinitialized
			// or if src/v/modpath was removed but not src/v/cache.
			fmt.Fprintf(os.Stderr, "vgo: extracting %s %s\n", mod.Path, mod.Version)
		} else {
			if err := os.MkdirAll(filepath.Join(srcV, "cache", mod.Path, "@v"), 0777); err != nil {
				return "", err
			}
			fmt.Fprintf(os.Stderr, "vgo: downloading %s %s\n", mod.Path, mod.Version)
			if err := downloadZip(mod, zipfile); err != nil {
				return "", err
			}
		}
		if err := modfetch.Unzip(dir, zipfile, modpath, 0); err != nil {
			fmt.Fprintf(os.Stderr, "-> %s\n", err)
			return "", err
		}
	}
	checkModHash(mod)
	return dir, nil
}

func downloadZip(mod module.Version, target string) error {
	repo, err := modfetch.Lookup(mod.Path)
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
		z.Close()
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

var useModHash = false
var modHash map[module.Version][]string

func initModHash() {
	if modHash != nil {
		return
	}
	modHash = make(map[module.Version][]string)
	file := filepath.Join(ModRoot, "go.modverify")
	data, err := ioutil.ReadFile(file)
	if err != nil && os.IsNotExist(err) {
		return
	}
	if err != nil {
		base.Fatalf("vgo: %v", err)
	}
	useModHash = true
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
			base.Fatalf("vgo: malformed go.modverify:\n%s:%d: wrong number of fields %v", file, lineno, len(f))
		}
		mod := module.Version{Path: f[0], Version: f[1]}
		modHash[mod] = append(modHash[mod], f[2])
	}
}

func checkModHash(mod module.Version) {
	initModHash()
	if !useModHash {
		return
	}

	data, err := ioutil.ReadFile(filepath.Join(srcV, "cache", mod.Path, "@v", mod.Version+".ziphash"))
	if err != nil {
		base.Fatalf("vgo: verifying %s %s: %v", mod.Path, mod.Version, err)
	}
	h := strings.TrimSpace(string(data))
	if !strings.HasPrefix(h, "h1:") {
		base.Fatalf("vgo: verifying %s %s: unexpected ziphash: %q", mod.Path, mod.Version, h)
	}

	for _, vh := range modHash[mod] {
		if h == vh {
			return
		}
		if strings.HasPrefix(vh, "h1:") {
			base.Fatalf("vgo: verifying %s %s: module hash mismatch\n\tdownloaded:   %v\n\tgo.modverify: %v", mod.Path, mod.Version, h, vh)
		}
	}
	if len(modHash[mod]) > 0 {
		fmt.Fprintf(os.Stderr, "warning: verifying %s %s: unknown hashes in go.modverify: %v; adding %v", mod.Path, mod.Version, strings.Join(modHash[mod], ", "), h)
	}
	modHash[mod] = append(modHash[mod], h)
}

func findModHash(mod module.Version) string {
	data, err := ioutil.ReadFile(filepath.Join(srcV, "cache", mod.Path, "@v", mod.Version+".ziphash"))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

func writeModHash() {
	if !useModHash {
		return
	}

	var mods []module.Version
	for m := range modHash {
		mods = append(mods, m)
	}
	sortModules(mods)
	var buf bytes.Buffer
	for _, m := range mods {
		list := modHash[m]
		sort.Strings(list)
		for _, h := range list {
			fmt.Fprintf(&buf, "%s %s %s\n", m.Path, m.Version, h)
		}
	}

	file := filepath.Join(ModRoot, "go.modverify")
	data, _ := ioutil.ReadFile(filepath.Join(ModRoot, "go.modverify"))
	if bytes.Equal(data, buf.Bytes()) {
		return
	}

	if err := ioutil.WriteFile(file, buf.Bytes(), 0666); err != nil {
		base.Fatalf("vgo: writing go.modverify: %v", err)
	}
}

func sortModules(mods []module.Version) {
	sort.Slice(mods, func(i, j int) bool {
		mi := mods[i]
		mj := mods[j]
		if mi.Path != mj.Path {
			return mi.Path < mj.Path
		}
		return semver.Compare(mi.Version, mj.Version) < 0
	})
}
