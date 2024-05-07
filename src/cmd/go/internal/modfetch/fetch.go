// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/fsys"
	"cmd/go/internal/gover"
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/par"
	"cmd/go/internal/robustio"
	"cmd/go/internal/str"
	"cmd/go/internal/trace"

	"golang.org/x/mod/module"
	"golang.org/x/mod/sumdb/dirhash"
	modzip "golang.org/x/mod/zip"
)

var downloadCache par.ErrCache[module.Version, string] // version → directory

var ErrToolchain = errors.New("internal error: invalid operation on toolchain module")

// Download downloads the specific module version to the
// local download cache and returns the name of the directory
// corresponding to the root of the module's file tree.
func Download(ctx context.Context, mod module.Version) (dir string, err error) {
	if gover.IsToolchain(mod.Path) {
		return "", ErrToolchain
	}
	if err := checkCacheDir(ctx); err != nil {
		base.Fatal(err)
	}

	// The par.Cache here avoids duplicate work.
	return downloadCache.Do(mod, func() (string, error) {
		dir, err := download(ctx, mod)
		if err != nil {
			return "", err
		}
		checkMod(ctx, mod)

		// If go.mod exists (not an old legacy module), check version is not too new.
		if data, err := os.ReadFile(filepath.Join(dir, "go.mod")); err == nil {
			goVersion := gover.GoModLookup(data, "go")
			if gover.Compare(goVersion, gover.Local()) > 0 {
				return "", &gover.TooNewError{What: mod.String(), GoVersion: goVersion}
			}
		} else if !errors.Is(err, fs.ErrNotExist) {
			return "", err
		}

		return dir, nil
	})
}

func download(ctx context.Context, mod module.Version) (dir string, err error) {
	ctx, span := trace.StartSpan(ctx, "modfetch.download "+mod.String())
	defer span.Done()

	dir, err = DownloadDir(ctx, mod)
	if err == nil {
		// The directory has already been completely extracted (no .partial file exists).
		return dir, nil
	} else if dir == "" || !errors.Is(err, fs.ErrNotExist) {
		return "", err
	}

	// To avoid cluttering the cache with extraneous files,
	// DownloadZip uses the same lockfile as Download.
	// Invoke DownloadZip before locking the file.
	zipfile, err := DownloadZip(ctx, mod)
	if err != nil {
		return "", err
	}

	unlock, err := lockVersion(ctx, mod)
	if err != nil {
		return "", err
	}
	defer unlock()

	ctx, span = trace.StartSpan(ctx, "unzip "+zipfile)
	defer span.Done()

	// Check whether the directory was populated while we were waiting on the lock.
	_, dirErr := DownloadDir(ctx, mod)
	if dirErr == nil {
		return dir, nil
	}
	_, dirExists := dirErr.(*DownloadDirPartialError)

	// Clean up any remaining temporary directories created by old versions
	// (before 1.16), as well as partially extracted directories (indicated by
	// DownloadDirPartialError, usually because of a .partial file). This is only
	// safe to do because the lock file ensures that their writers are no longer
	// active.
	parentDir := filepath.Dir(dir)
	tmpPrefix := filepath.Base(dir) + ".tmp-"
	if old, err := filepath.Glob(filepath.Join(str.QuoteGlob(parentDir), str.QuoteGlob(tmpPrefix)+"*")); err == nil {
		for _, path := range old {
			RemoveAll(path) // best effort
		}
	}
	if dirExists {
		if err := RemoveAll(dir); err != nil {
			return "", err
		}
	}

	partialPath, err := CachePath(ctx, mod, "partial")
	if err != nil {
		return "", err
	}

	// Extract the module zip directory at its final location.
	//
	// To prevent other processes from reading the directory if we crash,
	// create a .partial file before extracting the directory, and delete
	// the .partial file afterward (all while holding the lock).
	//
	// Before Go 1.16, we extracted to a temporary directory with a random name
	// then renamed it into place with os.Rename. On Windows, this failed with
	// ERROR_ACCESS_DENIED when another process (usually an anti-virus scanner)
	// opened files in the temporary directory.
	//
	// Go 1.14.2 and higher respect .partial files. Older versions may use
	// partially extracted directories. 'go mod verify' can detect this,
	// and 'go clean -modcache' can fix it.
	if err := os.MkdirAll(parentDir, 0777); err != nil {
		return "", err
	}
	if err := os.WriteFile(partialPath, nil, 0666); err != nil {
		return "", err
	}
	if err := modzip.Unzip(dir, mod, zipfile); err != nil {
		fmt.Fprintf(os.Stderr, "-> %s\n", err)
		if rmErr := RemoveAll(dir); rmErr == nil {
			os.Remove(partialPath)
		}
		return "", err
	}
	if err := os.Remove(partialPath); err != nil {
		return "", err
	}

	if !cfg.ModCacheRW {
		makeDirsReadOnly(dir)
	}
	return dir, nil
}

var downloadZipCache par.ErrCache[module.Version, string]

// DownloadZip downloads the specific module version to the
// local zip cache and returns the name of the zip file.
func DownloadZip(ctx context.Context, mod module.Version) (zipfile string, err error) {
	// The par.Cache here avoids duplicate work.
	return downloadZipCache.Do(mod, func() (string, error) {
		zipfile, err := CachePath(ctx, mod, "zip")
		if err != nil {
			return "", err
		}
		ziphashfile := zipfile + "hash"

		// Return without locking if the zip and ziphash files exist.
		if _, err := os.Stat(zipfile); err == nil {
			if _, err := os.Stat(ziphashfile); err == nil {
				return zipfile, nil
			}
		}

		// The zip or ziphash file does not exist. Acquire the lock and create them.
		if cfg.CmdName != "mod download" {
			vers := mod.Version
			if mod.Path == "golang.org/toolchain" {
				// Shorten v0.0.1-go1.13.1.darwin-amd64 to go1.13.1.darwin-amd64
				_, vers, _ = strings.Cut(vers, "-")
				if i := strings.LastIndex(vers, "."); i >= 0 {
					goos, goarch, _ := strings.Cut(vers[i+1:], "-")
					vers = vers[:i] + " (" + goos + "/" + goarch + ")"
				}
				fmt.Fprintf(os.Stderr, "go: downloading %s\n", vers)
			} else {
				fmt.Fprintf(os.Stderr, "go: downloading %s %s\n", mod.Path, vers)
			}
		}
		unlock, err := lockVersion(ctx, mod)
		if err != nil {
			return "", err
		}
		defer unlock()

		if err := downloadZip(ctx, mod, zipfile); err != nil {
			return "", err
		}
		return zipfile, nil
	})
}

func downloadZip(ctx context.Context, mod module.Version, zipfile string) (err error) {
	ctx, span := trace.StartSpan(ctx, "modfetch.downloadZip "+zipfile)
	defer span.Done()

	// Double-check that the zipfile was not created while we were waiting for
	// the lock in DownloadZip.
	ziphashfile := zipfile + "hash"
	var zipExists, ziphashExists bool
	if _, err := os.Stat(zipfile); err == nil {
		zipExists = true
	}
	if _, err := os.Stat(ziphashfile); err == nil {
		ziphashExists = true
	}
	if zipExists && ziphashExists {
		return nil
	}

	// Create parent directories.
	if err := os.MkdirAll(filepath.Dir(zipfile), 0777); err != nil {
		return err
	}

	// Clean up any remaining tempfiles from previous runs.
	// This is only safe to do because the lock file ensures that their
	// writers are no longer active.
	tmpPattern := filepath.Base(zipfile) + "*.tmp"
	if old, err := filepath.Glob(filepath.Join(str.QuoteGlob(filepath.Dir(zipfile)), tmpPattern)); err == nil {
		for _, path := range old {
			os.Remove(path) // best effort
		}
	}

	// If the zip file exists, the ziphash file must have been deleted
	// or lost after a file system crash. Re-hash the zip without downloading.
	if zipExists {
		return hashZip(mod, zipfile, ziphashfile)
	}

	// From here to the os.Rename call below is functionally almost equivalent to
	// renameio.WriteToFile, with one key difference: we want to validate the
	// contents of the file (by hashing it) before we commit it. Because the file
	// is zip-compressed, we need an actual file — or at least an io.ReaderAt — to
	// validate it: we can't just tee the stream as we write it.
	f, err := tempFile(ctx, filepath.Dir(zipfile), filepath.Base(zipfile), 0666)
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			f.Close()
			os.Remove(f.Name())
		}
	}()

	var unrecoverableErr error
	err = TryProxies(func(proxy string) error {
		if unrecoverableErr != nil {
			return unrecoverableErr
		}
		repo := Lookup(ctx, proxy, mod.Path)
		err := repo.Zip(ctx, f, mod.Version)
		if err != nil {
			// Zip may have partially written to f before failing.
			// (Perhaps the server crashed while sending the file?)
			// Since we allow fallback on error in some cases, we need to fix up the
			// file to be empty again for the next attempt.
			if _, err := f.Seek(0, io.SeekStart); err != nil {
				unrecoverableErr = err
				return err
			}
			if err := f.Truncate(0); err != nil {
				unrecoverableErr = err
				return err
			}
		}
		return err
	})
	if err != nil {
		return err
	}

	// Double-check that the paths within the zip file are well-formed.
	//
	// TODO(bcmills): There is a similar check within the Unzip function. Can we eliminate one?
	fi, err := f.Stat()
	if err != nil {
		return err
	}
	z, err := zip.NewReader(f, fi.Size())
	if err != nil {
		return err
	}
	prefix := mod.Path + "@" + mod.Version + "/"
	for _, f := range z.File {
		if !strings.HasPrefix(f.Name, prefix) {
			return fmt.Errorf("zip for %s has unexpected file %s", prefix[:len(prefix)-1], f.Name)
		}
	}

	if err := f.Close(); err != nil {
		return err
	}

	// Hash the zip file and check the sum before renaming to the final location.
	if err := hashZip(mod, f.Name(), ziphashfile); err != nil {
		return err
	}
	if err := os.Rename(f.Name(), zipfile); err != nil {
		return err
	}

	// TODO(bcmills): Should we make the .zip and .ziphash files read-only to discourage tampering?

	return nil
}

// hashZip reads the zip file opened in f, then writes the hash to ziphashfile,
// overwriting that file if it exists.
//
// If the hash does not match go.sum (or the sumdb if enabled), hashZip returns
// an error and does not write ziphashfile.
func hashZip(mod module.Version, zipfile, ziphashfile string) (err error) {
	hash, err := dirhash.HashZip(zipfile, dirhash.DefaultHash)
	if err != nil {
		return err
	}
	if err := checkModSum(mod, hash); err != nil {
		return err
	}
	hf, err := lockedfile.Create(ziphashfile)
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := hf.Close(); err == nil && closeErr != nil {
			err = closeErr
		}
	}()
	if err := hf.Truncate(int64(len(hash))); err != nil {
		return err
	}
	if _, err := hf.WriteAt([]byte(hash), 0); err != nil {
		return err
	}
	return nil
}

// makeDirsReadOnly makes a best-effort attempt to remove write permissions for dir
// and its transitive contents.
func makeDirsReadOnly(dir string) {
	type pathMode struct {
		path string
		mode fs.FileMode
	}
	var dirs []pathMode // in lexical order
	filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err == nil && d.IsDir() {
			info, err := d.Info()
			if err == nil && info.Mode()&0222 != 0 {
				dirs = append(dirs, pathMode{path, info.Mode()})
			}
		}
		return nil
	})

	// Run over list backward to chmod children before parents.
	for i := len(dirs) - 1; i >= 0; i-- {
		os.Chmod(dirs[i].path, dirs[i].mode&^0222)
	}
}

// RemoveAll removes a directory written by Download or Unzip, first applying
// any permission changes needed to do so.
func RemoveAll(dir string) error {
	// Module cache has 0555 directories; make them writable in order to remove content.
	filepath.WalkDir(dir, func(path string, info fs.DirEntry, err error) error {
		if err != nil {
			return nil // ignore errors walking in file system
		}
		if info.IsDir() {
			os.Chmod(path, 0777)
		}
		return nil
	})
	return robustio.RemoveAll(dir)
}

var GoSumFile string             // path to go.sum; set by package modload
var WorkspaceGoSumFiles []string // path to module go.sums in workspace; set by package modload

type modSum struct {
	mod module.Version
	sum string
}

var goSum struct {
	mu        sync.Mutex
	m         map[module.Version][]string            // content of go.sum file
	w         map[string]map[module.Version][]string // sum file in workspace -> content of that sum file
	status    map[modSum]modSumStatus                // state of sums in m
	overwrite bool                                   // if true, overwrite go.sum without incorporating its contents
	enabled   bool                                   // whether to use go.sum at all
}

type modSumStatus struct {
	used, dirty bool
}

// Reset resets globals in the modfetch package, so previous loads don't affect
// contents of go.sum files.
func Reset() {
	GoSumFile = ""
	WorkspaceGoSumFiles = nil

	// Uses of lookupCache and downloadCache both can call checkModSum,
	// which in turn sets the used bit on goSum.status for modules.
	// Reset them so used can be computed properly.
	lookupCache = par.Cache[lookupCacheKey, Repo]{}
	downloadCache = par.ErrCache[module.Version, string]{}

	// Clear all fields on goSum. It will be initialized later
	goSum.mu.Lock()
	goSum.m = nil
	goSum.w = nil
	goSum.status = nil
	goSum.overwrite = false
	goSum.enabled = false
	goSum.mu.Unlock()
}

// initGoSum initializes the go.sum data.
// The boolean it returns reports whether the
// use of go.sum is now enabled.
// The goSum lock must be held.
func initGoSum() (bool, error) {
	if GoSumFile == "" {
		return false, nil
	}
	if goSum.m != nil {
		return true, nil
	}

	goSum.m = make(map[module.Version][]string)
	goSum.status = make(map[modSum]modSumStatus)
	goSum.w = make(map[string]map[module.Version][]string)

	for _, f := range WorkspaceGoSumFiles {
		goSum.w[f] = make(map[module.Version][]string)
		_, err := readGoSumFile(goSum.w[f], f)
		if err != nil {
			return false, err
		}
	}

	enabled, err := readGoSumFile(goSum.m, GoSumFile)
	goSum.enabled = enabled
	return enabled, err
}

func readGoSumFile(dst map[module.Version][]string, file string) (bool, error) {
	var (
		data []byte
		err  error
	)
	if actualSumFile, ok := fsys.OverlayPath(file); ok {
		// Don't lock go.sum if it's part of the overlay.
		// On Plan 9, locking requires chmod, and we don't want to modify any file
		// in the overlay. See #44700.
		data, err = os.ReadFile(actualSumFile)
	} else {
		data, err = lockedfile.Read(file)
	}
	if err != nil && !os.IsNotExist(err) {
		return false, err
	}
	readGoSum(dst, file, data)

	return true, nil
}

// emptyGoModHash is the hash of a 1-file tree containing a 0-length go.mod.
// A bug caused us to write these into go.sum files for non-modules.
// We detect and remove them.
const emptyGoModHash = "h1:G7mAYYxgmS0lVkHyy2hEOLQCFB0DlQFTMLWggykrydY="

// readGoSum parses data, which is the content of file,
// and adds it to goSum.m. The goSum lock must be held.
func readGoSum(dst map[module.Version][]string, file string, data []byte) {
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
			if cfg.CmdName == "mod tidy" {
				// ignore malformed line so that go mod tidy can fix go.sum
				continue
			} else {
				base.Fatalf("malformed go.sum:\n%s:%d: wrong number of fields %v\n", file, lineno, len(f))
			}
		}
		if f[2] == emptyGoModHash {
			// Old bug; drop it.
			continue
		}
		mod := module.Version{Path: f[0], Version: f[1]}
		dst[mod] = append(dst[mod], f[2])
	}
}

// HaveSum returns true if the go.sum file contains an entry for mod.
// The entry's hash must be generated with a known hash algorithm.
// mod.Version may have a "/go.mod" suffix to distinguish sums for
// .mod and .zip files.
func HaveSum(mod module.Version) bool {
	goSum.mu.Lock()
	defer goSum.mu.Unlock()
	inited, err := initGoSum()
	if err != nil || !inited {
		return false
	}
	for _, goSums := range goSum.w {
		for _, h := range goSums[mod] {
			if !strings.HasPrefix(h, "h1:") {
				continue
			}
			if !goSum.status[modSum{mod, h}].dirty {
				return true
			}
		}
	}
	for _, h := range goSum.m[mod] {
		if !strings.HasPrefix(h, "h1:") {
			continue
		}
		if !goSum.status[modSum{mod, h}].dirty {
			return true
		}
	}
	return false
}

// RecordedSum returns the sum if the go.sum file contains an entry for mod.
// The boolean reports true if an entry was found or
// false if no entry found or two conflicting sums are found.
// The entry's hash must be generated with a known hash algorithm.
// mod.Version may have a "/go.mod" suffix to distinguish sums for
// .mod and .zip files.
func RecordedSum(mod module.Version) (sum string, ok bool) {
	goSum.mu.Lock()
	defer goSum.mu.Unlock()
	inited, err := initGoSum()
	foundSum := ""
	if err != nil || !inited {
		return "", false
	}
	for _, goSums := range goSum.w {
		for _, h := range goSums[mod] {
			if !strings.HasPrefix(h, "h1:") {
				continue
			}
			if !goSum.status[modSum{mod, h}].dirty {
				if foundSum != "" && foundSum != h { // conflicting sums exist
					return "", false
				}
				foundSum = h
			}
		}
	}
	for _, h := range goSum.m[mod] {
		if !strings.HasPrefix(h, "h1:") {
			continue
		}
		if !goSum.status[modSum{mod, h}].dirty {
			if foundSum != "" && foundSum != h { // conflicting sums exist
				return "", false
			}
			foundSum = h
		}
	}
	return foundSum, true
}

// checkMod checks the given module's checksum and Go version.
func checkMod(ctx context.Context, mod module.Version) {
	// Do the file I/O before acquiring the go.sum lock.
	ziphash, err := CachePath(ctx, mod, "ziphash")
	if err != nil {
		base.Fatalf("verifying %v", module.VersionError(mod, err))
	}
	data, err := lockedfile.Read(ziphash)
	if err != nil {
		base.Fatalf("verifying %v", module.VersionError(mod, err))
	}
	data = bytes.TrimSpace(data)
	if !isValidSum(data) {
		// Recreate ziphash file from zip file and use that to check the mod sum.
		zip, err := CachePath(ctx, mod, "zip")
		if err != nil {
			base.Fatalf("verifying %v", module.VersionError(mod, err))
		}
		err = hashZip(mod, zip, ziphash)
		if err != nil {
			base.Fatalf("verifying %v", module.VersionError(mod, err))
		}
		return
	}
	h := string(data)
	if !strings.HasPrefix(h, "h1:") {
		base.Fatalf("verifying %v", module.VersionError(mod, fmt.Errorf("unexpected ziphash: %q", h)))
	}

	if err := checkModSum(mod, h); err != nil {
		base.Fatalf("%s", err)
	}
}

// goModSum returns the checksum for the go.mod contents.
func goModSum(data []byte) (string, error) {
	return dirhash.Hash1([]string{"go.mod"}, func(string) (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(data)), nil
	})
}

// checkGoMod checks the given module's go.mod checksum;
// data is the go.mod content.
func checkGoMod(path, version string, data []byte) error {
	h, err := goModSum(data)
	if err != nil {
		return &module.ModuleError{Path: path, Version: version, Err: fmt.Errorf("verifying go.mod: %v", err)}
	}

	return checkModSum(module.Version{Path: path, Version: version + "/go.mod"}, h)
}

// checkModSum checks that the recorded checksum for mod is h.
//
// mod.Version may have the additional suffix "/go.mod" to request the checksum
// for the module's go.mod file only.
func checkModSum(mod module.Version, h string) error {
	// We lock goSum when manipulating it,
	// but we arrange to release the lock when calling checkSumDB,
	// so that parallel calls to checkModHash can execute parallel calls
	// to checkSumDB.

	// Check whether mod+h is listed in go.sum already. If so, we're done.
	goSum.mu.Lock()
	inited, err := initGoSum()
	if err != nil {
		goSum.mu.Unlock()
		return err
	}
	done := inited && haveModSumLocked(mod, h)
	if inited {
		st := goSum.status[modSum{mod, h}]
		st.used = true
		goSum.status[modSum{mod, h}] = st
	}
	goSum.mu.Unlock()

	if done {
		return nil
	}

	// Not listed, so we want to add them.
	// Consult checksum database if appropriate.
	if useSumDB(mod) {
		// Calls base.Fatalf if mismatch detected.
		if err := checkSumDB(mod, h); err != nil {
			return err
		}
	}

	// Add mod+h to go.sum, if it hasn't appeared already.
	if inited {
		goSum.mu.Lock()
		addModSumLocked(mod, h)
		st := goSum.status[modSum{mod, h}]
		st.dirty = true
		goSum.status[modSum{mod, h}] = st
		goSum.mu.Unlock()
	}
	return nil
}

// haveModSumLocked reports whether the pair mod,h is already listed in go.sum.
// If it finds a conflicting pair instead, it calls base.Fatalf.
// goSum.mu must be locked.
func haveModSumLocked(mod module.Version, h string) bool {
	sumFileName := "go.sum"
	if strings.HasSuffix(GoSumFile, "go.work.sum") {
		sumFileName = "go.work.sum"
	}
	for _, vh := range goSum.m[mod] {
		if h == vh {
			return true
		}
		if strings.HasPrefix(vh, "h1:") {
			base.Fatalf("verifying %s@%s: checksum mismatch\n\tdownloaded: %v\n\t%s:     %v"+goSumMismatch, mod.Path, mod.Version, h, sumFileName, vh)
		}
	}
	// Also check workspace sums.
	foundMatch := false
	// Check sums from all files in case there are conflicts between
	// the files.
	for goSumFile, goSums := range goSum.w {
		for _, vh := range goSums[mod] {
			if h == vh {
				foundMatch = true
			} else if strings.HasPrefix(vh, "h1:") {
				base.Fatalf("verifying %s@%s: checksum mismatch\n\tdownloaded: %v\n\t%s:     %v"+goSumMismatch, mod.Path, mod.Version, h, goSumFile, vh)
			}
		}
	}
	return foundMatch
}

// addModSumLocked adds the pair mod,h to go.sum.
// goSum.mu must be locked.
func addModSumLocked(mod module.Version, h string) {
	if haveModSumLocked(mod, h) {
		return
	}
	if len(goSum.m[mod]) > 0 {
		fmt.Fprintf(os.Stderr, "warning: verifying %s@%s: unknown hashes in go.sum: %v; adding %v"+hashVersionMismatch, mod.Path, mod.Version, strings.Join(goSum.m[mod], ", "), h)
	}
	goSum.m[mod] = append(goSum.m[mod], h)
}

// checkSumDB checks the mod, h pair against the Go checksum database.
// It calls base.Fatalf if the hash is to be rejected.
func checkSumDB(mod module.Version, h string) error {
	modWithoutSuffix := mod
	noun := "module"
	if before, found := strings.CutSuffix(mod.Version, "/go.mod"); found {
		noun = "go.mod"
		modWithoutSuffix.Version = before
	}

	db, lines, err := lookupSumDB(mod)
	if err != nil {
		return module.VersionError(modWithoutSuffix, fmt.Errorf("verifying %s: %v", noun, err))
	}

	have := mod.Path + " " + mod.Version + " " + h
	prefix := mod.Path + " " + mod.Version + " h1:"
	for _, line := range lines {
		if line == have {
			return nil
		}
		if strings.HasPrefix(line, prefix) {
			return module.VersionError(modWithoutSuffix, fmt.Errorf("verifying %s: checksum mismatch\n\tdownloaded: %v\n\t%s: %v"+sumdbMismatch, noun, h, db, line[len(prefix)-len("h1:"):]))
		}
	}
	return nil
}

// Sum returns the checksum for the downloaded copy of the given module,
// if present in the download cache.
func Sum(ctx context.Context, mod module.Version) string {
	if cfg.GOMODCACHE == "" {
		// Do not use current directory.
		return ""
	}

	ziphash, err := CachePath(ctx, mod, "ziphash")
	if err != nil {
		return ""
	}
	data, err := lockedfile.Read(ziphash)
	if err != nil {
		return ""
	}
	data = bytes.TrimSpace(data)
	if !isValidSum(data) {
		return ""
	}
	return string(data)
}

// isValidSum returns true if data is the valid contents of a zip hash file.
// Certain critical files are written to disk by first truncating
// then writing the actual bytes, so that if the write fails
// the corrupt file should contain at least one of the null
// bytes written by the truncate operation.
func isValidSum(data []byte) bool {
	if bytes.IndexByte(data, '\000') >= 0 {
		return false
	}

	if len(data) != len("h1:")+base64.StdEncoding.EncodedLen(sha256.Size) {
		return false
	}

	return true
}

var ErrGoSumDirty = errors.New("updates to go.sum needed, disabled by -mod=readonly")

// WriteGoSum writes the go.sum file if it needs to be updated.
//
// keep is used to check whether a newly added sum should be saved in go.sum.
// It should have entries for both module content sums and go.mod sums
// (version ends with "/go.mod"). Existing sums will be preserved unless they
// have been marked for deletion with TrimGoSum.
func WriteGoSum(ctx context.Context, keep map[module.Version]bool, readonly bool) error {
	goSum.mu.Lock()
	defer goSum.mu.Unlock()

	// If we haven't read the go.sum file yet, don't bother writing it.
	if !goSum.enabled {
		return nil
	}

	// Check whether we need to add sums for which keep[m] is true or remove
	// unused sums marked with TrimGoSum. If there are no changes to make,
	// just return without opening go.sum.
	dirty := false
Outer:
	for m, hs := range goSum.m {
		for _, h := range hs {
			st := goSum.status[modSum{m, h}]
			if st.dirty && (!st.used || keep[m]) {
				dirty = true
				break Outer
			}
		}
	}
	if !dirty {
		return nil
	}
	if readonly {
		return ErrGoSumDirty
	}
	if _, ok := fsys.OverlayPath(GoSumFile); ok {
		base.Fatalf("go: updates to go.sum needed, but go.sum is part of the overlay specified with -overlay")
	}

	// Make a best-effort attempt to acquire the side lock, only to exclude
	// previous versions of the 'go' command from making simultaneous edits.
	if unlock, err := SideLock(ctx); err == nil {
		defer unlock()
	}

	err := lockedfile.Transform(GoSumFile, func(data []byte) ([]byte, error) {
		tidyGoSum := tidyGoSum(data, keep)
		return tidyGoSum, nil
	})

	if err != nil {
		return fmt.Errorf("updating go.sum: %w", err)
	}

	goSum.status = make(map[modSum]modSumStatus)
	goSum.overwrite = false
	return nil
}

// TidyGoSum returns a tidy version of the go.sum file.
// A missing go.sum file is treated as if empty.
func TidyGoSum(keep map[module.Version]bool) (before, after []byte) {
	goSum.mu.Lock()
	defer goSum.mu.Unlock()
	before, err := lockedfile.Read(GoSumFile)
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		base.Fatalf("reading go.sum: %v", err)
	}
	after = tidyGoSum(before, keep)
	return before, after
}

// tidyGoSum will return a tidy version of the go.sum file.
// The goSum lock must be held.
func tidyGoSum(data []byte, keep map[module.Version]bool) []byte {
	if !goSum.overwrite {
		// Incorporate any sums added by other processes in the meantime.
		// Add only the sums that we actually checked: the user may have edited or
		// truncated the file to remove erroneous hashes, and we shouldn't restore
		// them without good reason.
		goSum.m = make(map[module.Version][]string, len(goSum.m))
		readGoSum(goSum.m, GoSumFile, data)
		for ms, st := range goSum.status {
			if st.used && !sumInWorkspaceModulesLocked(ms.mod) {
				addModSumLocked(ms.mod, ms.sum)
			}
		}
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
		str.Uniq(&list)
		for _, h := range list {
			st := goSum.status[modSum{m, h}]
			if (!st.dirty || (st.used && keep[m])) && !sumInWorkspaceModulesLocked(m) {
				fmt.Fprintf(&buf, "%s %s %s\n", m.Path, m.Version, h)
			}
		}
	}
	return buf.Bytes()
}

func sumInWorkspaceModulesLocked(m module.Version) bool {
	for _, goSums := range goSum.w {
		if _, ok := goSums[m]; ok {
			return true
		}
	}
	return false
}

// TrimGoSum trims go.sum to contain only the modules needed for reproducible
// builds.
//
// keep is used to check whether a sum should be retained in go.mod. It should
// have entries for both module content sums and go.mod sums (version ends
// with "/go.mod").
func TrimGoSum(keep map[module.Version]bool) {
	goSum.mu.Lock()
	defer goSum.mu.Unlock()
	inited, err := initGoSum()
	if err != nil {
		base.Fatalf("%s", err)
	}
	if !inited {
		return
	}

	for m, hs := range goSum.m {
		if !keep[m] {
			for _, h := range hs {
				goSum.status[modSum{m, h}] = modSumStatus{used: false, dirty: true}
			}
			goSum.overwrite = true
		}
	}
}

const goSumMismatch = `

SECURITY ERROR
This download does NOT match an earlier download recorded in go.sum.
The bits may have been replaced on the origin server, or an attacker may
have intercepted the download attempt.

For more information, see 'go help module-auth'.
`

const sumdbMismatch = `

SECURITY ERROR
This download does NOT match the one reported by the checksum server.
The bits may have been replaced on the origin server, or an attacker may
have intercepted the download attempt.

For more information, see 'go help module-auth'.
`

const hashVersionMismatch = `

SECURITY WARNING
This download is listed in go.sum, but using an unknown hash algorithm.
The download cannot be verified.

For more information, see 'go help module-auth'.

`

var HelpModuleAuth = &base.Command{
	UsageLine: "module-auth",
	Short:     "module authentication using go.sum",
	Long: `
When the go command downloads a module zip file or go.mod file into the
module cache, it computes a cryptographic hash and compares it with a known
value to verify the file hasn't changed since it was first downloaded. Known
hashes are stored in a file in the module root directory named go.sum. Hashes
may also be downloaded from the checksum database depending on the values of
GOSUMDB, GOPRIVATE, and GONOSUMDB.

For details, see https://golang.org/ref/mod#authenticating.
`,
}

var HelpPrivate = &base.Command{
	UsageLine: "private",
	Short:     "configuration for downloading non-public code",
	Long: `
The go command defaults to downloading modules from the public Go module
mirror at proxy.golang.org. It also defaults to validating downloaded modules,
regardless of source, against the public Go checksum database at sum.golang.org.
These defaults work well for publicly available source code.

The GOPRIVATE environment variable controls which modules the go command
considers to be private (not available publicly) and should therefore not use
the proxy or checksum database. The variable is a comma-separated list of
glob patterns (in the syntax of Go's path.Match) of module path prefixes.
For example,

	GOPRIVATE=*.corp.example.com,rsc.io/private

causes the go command to treat as private any module with a path prefix
matching either pattern, including git.corp.example.com/xyzzy, rsc.io/private,
and rsc.io/private/quux.

For fine-grained control over module download and validation, the GONOPROXY
and GONOSUMDB environment variables accept the same kind of glob list
and override GOPRIVATE for the specific decision of whether to use the proxy
and checksum database, respectively.

For example, if a company ran a module proxy serving private modules,
users would configure go using:

	GOPRIVATE=*.corp.example.com
	GOPROXY=proxy.example.com
	GONOPROXY=none

The GOPRIVATE variable is also used to define the "public" and "private"
patterns for the GOVCS variable; see 'go help vcs'. For that usage,
GOPRIVATE applies even in GOPATH mode. In that case, it matches import paths
instead of module paths.

The 'go env -w' command (see 'go help env') can be used to set these variables
for future go command invocations.

For more details, see https://golang.org/ref/mod#private-modules.
`,
}
