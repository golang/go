// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"archive/zip"
	"bytes"
	"errors"
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
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/par"
	"cmd/go/internal/renameio"
	"cmd/go/internal/robustio"

	"golang.org/x/mod/module"
	"golang.org/x/mod/sumdb/dirhash"
	modzip "golang.org/x/mod/zip"
)

var downloadCache par.Cache

// Download downloads the specific module version to the
// local download cache and returns the name of the directory
// corresponding to the root of the module's file tree.
func Download(mod module.Version) (dir string, err error) {
	if cfg.GOMODCACHE == "" {
		// modload.Init exits if GOPATH[0] is empty, and cfg.GOMODCACHE
		// is set to GOPATH[0]/pkg/mod if GOMODCACHE is empty, so this should never happen.
		base.Fatalf("go: internal error: cfg.GOMODCACHE not set")
	}

	// The par.Cache here avoids duplicate work.
	type cached struct {
		dir string
		err error
	}
	c := downloadCache.Do(mod, func() interface{} {
		dir, err := download(mod)
		if err != nil {
			return cached{"", err}
		}
		checkMod(mod)
		return cached{dir, nil}
	}).(cached)
	return c.dir, c.err
}

func download(mod module.Version) (dir string, err error) {
	// If the directory exists, and no .partial file exists, the module has
	// already been completely extracted. .partial files may be created when a
	// module zip directory is extracted in place instead of being extracted to a
	// temporary directory and renamed.
	dir, err = DownloadDir(mod)
	if err == nil {
		return dir, nil
	} else if dir == "" || !errors.Is(err, os.ErrNotExist) {
		return "", err
	}

	// To avoid cluttering the cache with extraneous files,
	// DownloadZip uses the same lockfile as Download.
	// Invoke DownloadZip before locking the file.
	zipfile, err := DownloadZip(mod)
	if err != nil {
		return "", err
	}

	unlock, err := lockVersion(mod)
	if err != nil {
		return "", err
	}
	defer unlock()

	// Check whether the directory was populated while we were waiting on the lock.
	_, dirErr := DownloadDir(mod)
	if dirErr == nil {
		return dir, nil
	}
	_, dirExists := dirErr.(*DownloadDirPartialError)

	// Clean up any remaining temporary directories from previous runs, as well
	// as partially extracted diectories created by future versions of cmd/go.
	// This is only safe to do because the lock file ensures that their writers
	// are no longer active.
	parentDir := filepath.Dir(dir)
	tmpPrefix := filepath.Base(dir) + ".tmp-"
	if old, err := filepath.Glob(filepath.Join(parentDir, tmpPrefix+"*")); err == nil {
		for _, path := range old {
			RemoveAll(path) // best effort
		}
	}
	if dirExists {
		if err := RemoveAll(dir); err != nil {
			return "", err
		}
	}

	partialPath, err := CachePath(mod, "partial")
	if err != nil {
		return "", err
	}
	if err := os.Remove(partialPath); err != nil && !os.IsNotExist(err) {
		return "", err
	}

	// Extract the module zip directory.
	//
	// By default, we extract to a temporary directory, then atomically rename to
	// its final location. We use the existence of the source directory to signal
	// that it has been extracted successfully (see DownloadDir).  If someone
	// deletes the entire directory (e.g., as an attempt to prune out file
	// corruption), the module cache will still be left in a recoverable
	// state.
	//
	// Unfortunately, os.Rename may fail with ERROR_ACCESS_DENIED on Windows if
	// another process opens files in the temporary directory. This is partially
	// mitigated by using robustio.Rename, which retries os.Rename for a short
	// time.
	//
	// To avoid this error completely, if unzipInPlace is set, we instead create a
	// .partial file (indicating the directory isn't fully extracted), then we
	// extract the directory at its final location, then we delete the .partial
	// file. This is not the default behavior because older versions of Go may
	// simply stat the directory to check whether it exists without looking for a
	// .partial file. If multiple versions run concurrently, the older version may
	// assume a partially extracted directory is complete.
	// TODO(golang.org/issue/36568): when these older versions are no longer
	// supported, remove the old default behavior and the unzipInPlace flag.
	if err := os.MkdirAll(parentDir, 0777); err != nil {
		return "", err
	}

	if unzipInPlace {
		if err := ioutil.WriteFile(partialPath, nil, 0666); err != nil {
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
	} else {
		tmpDir, err := ioutil.TempDir(parentDir, tmpPrefix)
		if err != nil {
			return "", err
		}
		if err := modzip.Unzip(tmpDir, mod, zipfile); err != nil {
			fmt.Fprintf(os.Stderr, "-> %s\n", err)
			RemoveAll(tmpDir)
			return "", err
		}
		if err := robustio.Rename(tmpDir, dir); err != nil {
			RemoveAll(tmpDir)
			return "", err
		}
	}

	if !cfg.ModCacheRW {
		// Make dir read-only only *after* renaming it.
		// os.Rename was observed to fail for read-only directories on macOS.
		makeDirsReadOnly(dir)
	}
	return dir, nil
}

var unzipInPlace bool

func init() {
	for _, f := range strings.Split(os.Getenv("GODEBUG"), ",") {
		if f == "modcacheunzipinplace=1" {
			unzipInPlace = true
			break
		}
	}
}

var downloadZipCache par.Cache

// DownloadZip downloads the specific module version to the
// local zip cache and returns the name of the zip file.
func DownloadZip(mod module.Version) (zipfile string, err error) {
	// The par.Cache here avoids duplicate work.
	type cached struct {
		zipfile string
		err     error
	}
	c := downloadZipCache.Do(mod, func() interface{} {
		zipfile, err := CachePath(mod, "zip")
		if err != nil {
			return cached{"", err}
		}

		// Skip locking if the zipfile already exists.
		if _, err := os.Stat(zipfile); err == nil {
			return cached{zipfile, nil}
		}

		// The zip file does not exist. Acquire the lock and create it.
		if cfg.CmdName != "mod download" {
			fmt.Fprintf(os.Stderr, "go: downloading %s %s\n", mod.Path, mod.Version)
		}
		unlock, err := lockVersion(mod)
		if err != nil {
			return cached{"", err}
		}
		defer unlock()

		// Double-check that the zipfile was not created while we were waiting for
		// the lock.
		if _, err := os.Stat(zipfile); err == nil {
			return cached{zipfile, nil}
		}
		if err := os.MkdirAll(filepath.Dir(zipfile), 0777); err != nil {
			return cached{"", err}
		}
		if err := downloadZip(mod, zipfile); err != nil {
			return cached{"", err}
		}
		return cached{zipfile, nil}
	}).(cached)
	return c.zipfile, c.err
}

func downloadZip(mod module.Version, zipfile string) (err error) {
	// Clean up any remaining tempfiles from previous runs.
	// This is only safe to do because the lock file ensures that their
	// writers are no longer active.
	for _, base := range []string{zipfile, zipfile + "hash"} {
		if old, err := filepath.Glob(renameio.Pattern(base)); err == nil {
			for _, path := range old {
				os.Remove(path) // best effort
			}
		}
	}

	// From here to the os.Rename call below is functionally almost equivalent to
	// renameio.WriteToFile, with one key difference: we want to validate the
	// contents of the file (by hashing it) before we commit it. Because the file
	// is zip-compressed, we need an actual file — or at least an io.ReaderAt — to
	// validate it: we can't just tee the stream as we write it.
	f, err := ioutil.TempFile(filepath.Dir(zipfile), filepath.Base(renameio.Pattern(zipfile)))
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			f.Close()
			os.Remove(f.Name())
		}
	}()

	err = TryProxies(func(proxy string) error {
		repo, err := Lookup(proxy, mod.Path)
		if err != nil {
			return err
		}
		return repo.Zip(f, mod.Version)
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

	// Sync the file before renaming it: otherwise, after a crash the reader may
	// observe a 0-length file instead of the actual contents.
	// See https://golang.org/issue/22397#issuecomment-380831736.
	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}

	// Hash the zip file and check the sum before renaming to the final location.
	hash, err := dirhash.HashZip(f.Name(), dirhash.DefaultHash)
	if err != nil {
		return err
	}
	if err := checkModSum(mod, hash); err != nil {
		return err
	}

	if err := renameio.WriteFile(zipfile+"hash", []byte(hash), 0666); err != nil {
		return err
	}
	if err := os.Rename(f.Name(), zipfile); err != nil {
		return err
	}

	// TODO(bcmills): Should we make the .zip and .ziphash files read-only to discourage tampering?

	return nil
}

// makeDirsReadOnly makes a best-effort attempt to remove write permissions for dir
// and its transitive contents.
func makeDirsReadOnly(dir string) {
	type pathMode struct {
		path string
		mode os.FileMode
	}
	var dirs []pathMode // in lexical order
	filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err == nil && info.Mode()&0222 != 0 {
			if info.IsDir() {
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
	filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
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

var GoSumFile string // path to go.sum; set by package modload

type modSum struct {
	mod module.Version
	sum string
}

var goSum struct {
	mu        sync.Mutex
	m         map[module.Version][]string // content of go.sum file (+ go.modverify if present)
	checked   map[modSum]bool             // sums actually checked during execution
	dirty     bool                        // whether we added any new sums to m
	overwrite bool                        // if true, overwrite go.sum without incorporating its contents
	enabled   bool                        // whether to use go.sum at all
	modverify string                      // path to go.modverify, to be deleted
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
	goSum.checked = make(map[modSum]bool)
	data, err := lockedfile.Read(GoSumFile)
	if err != nil && !os.IsNotExist(err) {
		return false, err
	}
	goSum.enabled = true
	readGoSum(goSum.m, GoSumFile, data)

	// Add old go.modverify file.
	// We'll delete go.modverify in WriteGoSum.
	alt := strings.TrimSuffix(GoSumFile, ".sum") + ".modverify"
	if data, err := renameio.ReadFile(alt); err == nil {
		migrate := make(map[module.Version][]string)
		readGoSum(migrate, alt, data)
		for mod, sums := range migrate {
			for _, sum := range sums {
				addModSumLocked(mod, sum)
			}
		}
		goSum.modverify = alt
	}
	return true, nil
}

// emptyGoModHash is the hash of a 1-file tree containing a 0-length go.mod.
// A bug caused us to write these into go.sum files for non-modules.
// We detect and remove them.
const emptyGoModHash = "h1:G7mAYYxgmS0lVkHyy2hEOLQCFB0DlQFTMLWggykrydY="

// readGoSum parses data, which is the content of file,
// and adds it to goSum.m. The goSum lock must be held.
func readGoSum(dst map[module.Version][]string, file string, data []byte) error {
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
			return fmt.Errorf("malformed go.sum:\n%s:%d: wrong number of fields %v", file, lineno, len(f))
		}
		if f[2] == emptyGoModHash {
			// Old bug; drop it.
			continue
		}
		mod := module.Version{Path: f[0], Version: f[1]}
		dst[mod] = append(dst[mod], f[2])
	}
	return nil
}

// checkMod checks the given module's checksum.
func checkMod(mod module.Version) {
	if cfg.GOMODCACHE == "" {
		// Do not use current directory.
		return
	}

	// Do the file I/O before acquiring the go.sum lock.
	ziphash, err := CachePath(mod, "ziphash")
	if err != nil {
		base.Fatalf("verifying %v", module.VersionError(mod, err))
	}
	data, err := renameio.ReadFile(ziphash)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			// This can happen if someone does rm -rf GOPATH/src/cache/download. So it goes.
			return
		}
		base.Fatalf("verifying %v", module.VersionError(mod, err))
	}
	h := strings.TrimSpace(string(data))
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
		return ioutil.NopCloser(bytes.NewReader(data)), nil
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
		goSum.mu.Unlock()
	}
	return nil
}

// haveModSumLocked reports whether the pair mod,h is already listed in go.sum.
// If it finds a conflicting pair instead, it calls base.Fatalf.
// goSum.mu must be locked.
func haveModSumLocked(mod module.Version, h string) bool {
	goSum.checked[modSum{mod, h}] = true
	for _, vh := range goSum.m[mod] {
		if h == vh {
			return true
		}
		if strings.HasPrefix(vh, "h1:") {
			base.Fatalf("verifying %s@%s: checksum mismatch\n\tdownloaded: %v\n\tgo.sum:     %v"+goSumMismatch, mod.Path, mod.Version, h, vh)
		}
	}
	return false
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
	goSum.dirty = true
}

// checkSumDB checks the mod, h pair against the Go checksum database.
// It calls base.Fatalf if the hash is to be rejected.
func checkSumDB(mod module.Version, h string) error {
	db, lines, err := lookupSumDB(mod)
	if err != nil {
		return module.VersionError(mod, fmt.Errorf("verifying module: %v", err))
	}

	have := mod.Path + " " + mod.Version + " " + h
	prefix := mod.Path + " " + mod.Version + " h1:"
	for _, line := range lines {
		if line == have {
			return nil
		}
		if strings.HasPrefix(line, prefix) {
			return module.VersionError(mod, fmt.Errorf("verifying module: checksum mismatch\n\tdownloaded: %v\n\t%s: %v"+sumdbMismatch, h, db, line[len(prefix)-len("h1:"):]))
		}
	}
	return nil
}

// Sum returns the checksum for the downloaded copy of the given module,
// if present in the download cache.
func Sum(mod module.Version) string {
	if cfg.GOMODCACHE == "" {
		// Do not use current directory.
		return ""
	}

	ziphash, err := CachePath(mod, "ziphash")
	if err != nil {
		return ""
	}
	data, err := renameio.ReadFile(ziphash)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

// WriteGoSum writes the go.sum file if it needs to be updated.
func WriteGoSum() {
	goSum.mu.Lock()
	defer goSum.mu.Unlock()

	if !goSum.enabled {
		// If we haven't read the go.sum file yet, don't bother writing it: at best,
		// we could rename the go.modverify file if it isn't empty, but we haven't
		// needed to touch it so far — how important could it be?
		return
	}
	if !goSum.dirty {
		// Don't bother opening the go.sum file if we don't have anything to add.
		return
	}
	if cfg.BuildMod == "readonly" {
		base.Fatalf("go: updates to go.sum needed, disabled by -mod=readonly")
	}

	// Make a best-effort attempt to acquire the side lock, only to exclude
	// previous versions of the 'go' command from making simultaneous edits.
	if unlock, err := SideLock(); err == nil {
		defer unlock()
	}

	err := lockedfile.Transform(GoSumFile, func(data []byte) ([]byte, error) {
		if !goSum.overwrite {
			// Incorporate any sums added by other processes in the meantime.
			// Add only the sums that we actually checked: the user may have edited or
			// truncated the file to remove erroneous hashes, and we shouldn't restore
			// them without good reason.
			goSum.m = make(map[module.Version][]string, len(goSum.m))
			readGoSum(goSum.m, GoSumFile, data)
			for ms := range goSum.checked {
				addModSumLocked(ms.mod, ms.sum)
				goSum.dirty = true
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
			for _, h := range list {
				fmt.Fprintf(&buf, "%s %s %s\n", m.Path, m.Version, h)
			}
		}
		return buf.Bytes(), nil
	})

	if err != nil {
		base.Fatalf("go: updating go.sum: %v", err)
	}

	goSum.checked = make(map[modSum]bool)
	goSum.dirty = false
	goSum.overwrite = false

	if goSum.modverify != "" {
		os.Remove(goSum.modverify) // best effort
	}
}

// TrimGoSum trims go.sum to contain only the modules for which keep[m] is true.
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

	for m := range goSum.m {
		// If we're keeping x@v we also keep x@v/go.mod.
		// Map x@v/go.mod back to x@v for the keep lookup.
		noGoMod := module.Version{Path: m.Path, Version: strings.TrimSuffix(m.Version, "/go.mod")}
		if !keep[m] && !keep[noGoMod] {
			delete(goSum.m, m)
			goSum.dirty = true
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
The go command tries to authenticate every downloaded module,
checking that the bits downloaded for a specific module version today
match bits downloaded yesterday. This ensures repeatable builds
and detects introduction of unexpected changes, malicious or not.

In each module's root, alongside go.mod, the go command maintains
a file named go.sum containing the cryptographic checksums of the
module's dependencies.

The form of each line in go.sum is three fields:

	<module> <version>[/go.mod] <hash>

Each known module version results in two lines in the go.sum file.
The first line gives the hash of the module version's file tree.
The second line appends "/go.mod" to the version and gives the hash
of only the module version's (possibly synthesized) go.mod file.
The go.mod-only hash allows downloading and authenticating a
module version's go.mod file, which is needed to compute the
dependency graph, without also downloading all the module's source code.

The hash begins with an algorithm prefix of the form "h<N>:".
The only defined algorithm prefix is "h1:", which uses SHA-256.

Module authentication failures

The go command maintains a cache of downloaded packages and computes
and records the cryptographic checksum of each package at download time.
In normal operation, the go command checks the main module's go.sum file
against these precomputed checksums instead of recomputing them on
each command invocation. The 'go mod verify' command checks that
the cached copies of module downloads still match both their recorded
checksums and the entries in go.sum.

In day-to-day development, the checksum of a given module version
should never change. Each time a dependency is used by a given main
module, the go command checks its local cached copy, freshly
downloaded or not, against the main module's go.sum. If the checksums
don't match, the go command reports the mismatch as a security error
and refuses to run the build. When this happens, proceed with caution:
code changing unexpectedly means today's build will not match
yesterday's, and the unexpected change may not be beneficial.

If the go command reports a mismatch in go.sum, the downloaded code
for the reported module version does not match the one used in a
previous build of the main module. It is important at that point
to find out what the right checksum should be, to decide whether
go.sum is wrong or the downloaded code is wrong. Usually go.sum is right:
you want to use the same code you used yesterday.

If a downloaded module is not yet included in go.sum and it is a publicly
available module, the go command consults the Go checksum database to fetch
the expected go.sum lines. If the downloaded code does not match those
lines, the go command reports the mismatch and exits. Note that the
database is not consulted for module versions already listed in go.sum.

If a go.sum mismatch is reported, it is always worth investigating why
the code downloaded today differs from what was downloaded yesterday.

The GOSUMDB environment variable identifies the name of checksum database
to use and optionally its public key and URL, as in:

	GOSUMDB="sum.golang.org"
	GOSUMDB="sum.golang.org+<publickey>"
	GOSUMDB="sum.golang.org+<publickey> https://sum.golang.org"

The go command knows the public key of sum.golang.org, and also that the name
sum.golang.google.cn (available inside mainland China) connects to the
sum.golang.org checksum database; use of any other database requires giving
the public key explicitly.
The URL defaults to "https://" followed by the database name.

GOSUMDB defaults to "sum.golang.org", the Go checksum database run by Google.
See https://sum.golang.org/privacy for the service's privacy policy.

If GOSUMDB is set to "off", or if "go get" is invoked with the -insecure flag,
the checksum database is not consulted, and all unrecognized modules are
accepted, at the cost of giving up the security guarantee of verified repeatable
downloads for all modules. A better way to bypass the checksum database
for specific modules is to use the GOPRIVATE or GONOSUMDB environment
variables. See 'go help module-private' for details.

The 'go env -w' command (see 'go help env') can be used to set these variables
for future go command invocations.
`,
}

var HelpModulePrivate = &base.Command{
	UsageLine: "module-private",
	Short:     "module configuration for non-public modules",
	Long: `
The go command defaults to downloading modules from the public Go module
mirror at proxy.golang.org. It also defaults to validating downloaded modules,
regardless of source, against the public Go checksum database at sum.golang.org.
These defaults work well for publicly available source code.

The GOPRIVATE environment variable controls which modules the go command
considers to be private (not available publicly) and should therefore not use the
proxy or checksum database. The variable is a comma-separated list of
glob patterns (in the syntax of Go's path.Match) of module path prefixes.
For example,

	GOPRIVATE=*.corp.example.com,rsc.io/private

causes the go command to treat as private any module with a path prefix
matching either pattern, including git.corp.example.com/xyzzy, rsc.io/private,
and rsc.io/private/quux.

The GOPRIVATE environment variable may be used by other tools as well to
identify non-public modules. For example, an editor could use GOPRIVATE
to decide whether to hyperlink a package import to a godoc.org page.

For fine-grained control over module download and validation, the GONOPROXY
and GONOSUMDB environment variables accept the same kind of glob list
and override GOPRIVATE for the specific decision of whether to use the proxy
and checksum database, respectively.

For example, if a company ran a module proxy serving private modules,
users would configure go using:

	GOPRIVATE=*.corp.example.com
	GOPROXY=proxy.example.com
	GONOPROXY=none

This would tell the go command and other tools that modules beginning with
a corp.example.com subdomain are private but that the company proxy should
be used for downloading both public and private modules, because
GONOPROXY has been set to a pattern that won't match any modules,
overriding GOPRIVATE.

The 'go env -w' command (see 'go help env') can be used to set these variables
for future go command invocations.
`,
}
