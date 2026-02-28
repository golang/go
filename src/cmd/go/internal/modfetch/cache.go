// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/gover"
	"cmd/go/internal/lockedfile"
	"cmd/go/internal/modfetch/codehost"
	"cmd/internal/par"
	"cmd/internal/robustio"
	"cmd/internal/telemetry/counter"

	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

func cacheDir(ctx context.Context, path string) (string, error) {
	if err := checkCacheDir(ctx); err != nil {
		return "", err
	}
	enc, err := module.EscapePath(path)
	if err != nil {
		return "", err
	}
	return filepath.Join(cfg.GOMODCACHE, "cache/download", enc, "/@v"), nil
}

func CachePath(ctx context.Context, m module.Version, suffix string) (string, error) {
	if gover.IsToolchain(m.Path) {
		return "", ErrToolchain
	}
	dir, err := cacheDir(ctx, m.Path)
	if err != nil {
		return "", err
	}
	if !gover.ModIsValid(m.Path, m.Version) {
		return "", fmt.Errorf("non-semver module version %q", m.Version)
	}
	if module.CanonicalVersion(m.Version) != m.Version {
		return "", fmt.Errorf("non-canonical module version %q", m.Version)
	}
	encVer, err := module.EscapeVersion(m.Version)
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, encVer+"."+suffix), nil
}

// DownloadDir returns the directory to which m should have been downloaded.
// An error will be returned if the module path or version cannot be escaped.
// An error satisfying errors.Is(err, fs.ErrNotExist) will be returned
// along with the directory if the directory does not exist or if the directory
// is not completely populated.
func DownloadDir(ctx context.Context, m module.Version) (string, error) {
	if gover.IsToolchain(m.Path) {
		return "", ErrToolchain
	}
	if err := checkCacheDir(ctx); err != nil {
		return "", err
	}
	enc, err := module.EscapePath(m.Path)
	if err != nil {
		return "", err
	}
	if !gover.ModIsValid(m.Path, m.Version) {
		return "", fmt.Errorf("non-semver module version %q", m.Version)
	}
	if module.CanonicalVersion(m.Version) != m.Version {
		return "", fmt.Errorf("non-canonical module version %q", m.Version)
	}
	encVer, err := module.EscapeVersion(m.Version)
	if err != nil {
		return "", err
	}

	// Check whether the directory itself exists.
	dir := filepath.Join(cfg.GOMODCACHE, enc+"@"+encVer)
	if fi, err := os.Stat(dir); os.IsNotExist(err) {
		return dir, err
	} else if err != nil {
		return dir, &DownloadDirPartialError{dir, err}
	} else if !fi.IsDir() {
		return dir, &DownloadDirPartialError{dir, errors.New("not a directory")}
	}

	// Check if a .partial file exists. This is created at the beginning of
	// a download and removed after the zip is extracted.
	partialPath, err := CachePath(ctx, m, "partial")
	if err != nil {
		return dir, err
	}
	if _, err := os.Stat(partialPath); err == nil {
		return dir, &DownloadDirPartialError{dir, errors.New("not completely extracted")}
	} else if !os.IsNotExist(err) {
		return dir, err
	}

	// Special case: ziphash is not required for the golang.org/fips140 module,
	// because it is unpacked from a file in GOROOT, not downloaded.
	// We've already checked that it's not a partial unpacking, so we're happy.
	if m.Path == "golang.org/fips140" {
		return dir, nil
	}

	// Check if a .ziphash file exists. It should be created before the
	// zip is extracted, but if it was deleted (by another program?), we need
	// to re-calculate it. Note that checkMod will repopulate the ziphash
	// file if it doesn't exist, but if the module is excluded by checks
	// through GONOSUMDB or GOPRIVATE, that check and repopulation won't happen.
	ziphashPath, err := CachePath(ctx, m, "ziphash")
	if err != nil {
		return dir, err
	}
	if _, err := os.Stat(ziphashPath); os.IsNotExist(err) {
		return dir, &DownloadDirPartialError{dir, errors.New("ziphash file is missing")}
	} else if err != nil {
		return dir, err
	}
	return dir, nil
}

// DownloadDirPartialError is returned by DownloadDir if a module directory
// exists but was not completely populated.
//
// DownloadDirPartialError is equivalent to fs.ErrNotExist.
type DownloadDirPartialError struct {
	Dir string
	Err error
}

func (e *DownloadDirPartialError) Error() string     { return fmt.Sprintf("%s: %v", e.Dir, e.Err) }
func (e *DownloadDirPartialError) Is(err error) bool { return err == fs.ErrNotExist }

// lockVersion locks a file within the module cache that guards the downloading
// and extraction of the zipfile for the given module version.
func lockVersion(ctx context.Context, mod module.Version) (unlock func(), err error) {
	path, err := CachePath(ctx, mod, "lock")
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o777); err != nil {
		return nil, err
	}
	return lockedfile.MutexAt(path).Lock()
}

// SideLock locks a file within the module cache that previously guarded
// edits to files outside the cache, such as go.sum and go.mod files in the
// user's working directory.
// If err is nil, the caller MUST eventually call the unlock function.
func SideLock(ctx context.Context) (unlock func(), err error) {
	if err := checkCacheDir(ctx); err != nil {
		return nil, err
	}

	path := filepath.Join(cfg.GOMODCACHE, "cache", "lock")
	if err := os.MkdirAll(filepath.Dir(path), 0o777); err != nil {
		return nil, fmt.Errorf("failed to create cache directory: %w", err)
	}

	return lockedfile.MutexAt(path).Lock()
}

// A cachingRepo is a cache around an underlying Repo,
// avoiding redundant calls to ModulePath, Versions, Stat, Latest, and GoMod (but not CheckReuse or Zip).
// It is also safe for simultaneous use by multiple goroutines
// (so that it can be returned from Lookup multiple times).
// It serializes calls to the underlying Repo.
type cachingRepo struct {
	path          string
	versionsCache par.ErrCache[string, *Versions]
	statCache     par.ErrCache[string, *RevInfo]
	latestCache   par.ErrCache[struct{}, *RevInfo]
	gomodCache    par.ErrCache[string, []byte]

	once     sync.Once
	initRepo func(context.Context) (Repo, error)
	r        Repo
	fetcher  *Fetcher
}

func newCachingRepo(ctx context.Context, fetcher *Fetcher, path string, initRepo func(context.Context) (Repo, error)) *cachingRepo {
	return &cachingRepo{
		path:     path,
		initRepo: initRepo,
		fetcher:  fetcher,
	}
}

func (r *cachingRepo) repo(ctx context.Context) Repo {
	r.once.Do(func() {
		var err error
		r.r, err = r.initRepo(ctx)
		if err != nil {
			r.r = errRepo{r.path, err}
		}
	})
	return r.r
}

func (r *cachingRepo) CheckReuse(ctx context.Context, old *codehost.Origin) error {
	return r.repo(ctx).CheckReuse(ctx, old)
}

func (r *cachingRepo) ModulePath() string {
	return r.path
}

func (r *cachingRepo) Versions(ctx context.Context, prefix string) (*Versions, error) {
	v, err := r.versionsCache.Do(prefix, func() (*Versions, error) {
		return r.repo(ctx).Versions(ctx, prefix)
	})
	if err != nil {
		return nil, err
	}
	return &Versions{
		Origin: v.Origin,
		List:   append([]string(nil), v.List...),
	}, nil
}

type cachedInfo struct {
	info *RevInfo
	err  error
}

func (r *cachingRepo) Stat(ctx context.Context, rev string) (*RevInfo, error) {
	if gover.IsToolchain(r.path) {
		// Skip disk cache; the underlying golang.org/toolchain repo is cached instead.
		return r.repo(ctx).Stat(ctx, rev)
	}
	info, err := r.statCache.Do(rev, func() (*RevInfo, error) {
		file, info, err := readDiskStat(ctx, r.path, rev)
		if err == nil {
			return info, err
		}

		info, err = r.repo(ctx).Stat(ctx, rev)
		if err == nil {
			// If we resolved, say, 1234abcde to v0.0.0-20180604122334-1234abcdef78,
			// then save the information under the proper version, for future use.
			if info.Version != rev {
				file, _ = CachePath(ctx, module.Version{Path: r.path, Version: info.Version}, "info")
				r.statCache.Do(info.Version, func() (*RevInfo, error) {
					return info, nil
				})
			}

			if err := writeDiskStat(ctx, file, info); err != nil {
				fmt.Fprintf(os.Stderr, "go: writing stat cache: %v\n", err)
			}
		}
		return info, err
	})
	if info != nil {
		copy := *info
		info = &copy
	}
	return info, err
}

func (r *cachingRepo) Latest(ctx context.Context) (*RevInfo, error) {
	if gover.IsToolchain(r.path) {
		// Skip disk cache; the underlying golang.org/toolchain repo is cached instead.
		return r.repo(ctx).Latest(ctx)
	}
	info, err := r.latestCache.Do(struct{}{}, func() (*RevInfo, error) {
		info, err := r.repo(ctx).Latest(ctx)

		// Save info for likely future Stat call.
		if err == nil {
			r.statCache.Do(info.Version, func() (*RevInfo, error) {
				return info, nil
			})
			if file, _, err := readDiskStat(ctx, r.path, info.Version); err != nil {
				writeDiskStat(ctx, file, info)
			}
		}

		return info, err
	})
	if info != nil {
		copy := *info
		info = &copy
	}
	return info, err
}

func (r *cachingRepo) GoMod(ctx context.Context, version string) ([]byte, error) {
	if gover.IsToolchain(r.path) {
		// Skip disk cache; the underlying golang.org/toolchain repo is cached instead.
		return r.repo(ctx).GoMod(ctx, version)
	}
	text, err := r.gomodCache.Do(version, func() ([]byte, error) {
		file, text, err := r.fetcher.readDiskGoMod(ctx, r.path, version)
		if err == nil {
			// Note: readDiskGoMod already called checkGoMod.
			return text, nil
		}

		text, err = r.repo(ctx).GoMod(ctx, version)
		if err == nil {
			if err := checkGoMod(r.fetcher, r.path, version, text); err != nil {
				return text, err
			}
			if err := writeDiskGoMod(ctx, file, text); err != nil {
				fmt.Fprintf(os.Stderr, "go: writing go.mod cache: %v\n", err)
			}
		}
		return text, err
	})
	if err != nil {
		return nil, err
	}
	return append([]byte(nil), text...), nil
}

func (r *cachingRepo) Zip(ctx context.Context, dst io.Writer, version string) error {
	if gover.IsToolchain(r.path) {
		return ErrToolchain
	}
	return r.repo(ctx).Zip(ctx, dst, version)
}

// InfoFile is like Lookup(ctx, path).Stat(version) but also returns the name of the file
// containing the cached information.
func (f *Fetcher) InfoFile(ctx context.Context, path, version string) (*RevInfo, string, error) {
	if !gover.ModIsValid(path, version) {
		return nil, "", fmt.Errorf("invalid version %q", version)
	}

	if file, info, err := readDiskStat(ctx, path, version); err == nil {
		return info, file, nil
	}

	var info *RevInfo
	var err2info map[error]*RevInfo
	err := TryProxies(func(proxy string) error {
		i, err := f.Lookup(ctx, proxy, path).Stat(ctx, version)
		if err == nil {
			info = i
		} else {
			if err2info == nil {
				err2info = make(map[error]*RevInfo)
			}
			err2info[err] = info
		}
		return err
	})
	if err != nil {
		return err2info[err], "", err
	}

	// Stat should have populated the disk cache for us.
	file, err := CachePath(ctx, module.Version{Path: path, Version: version}, "info")
	if err != nil {
		return nil, "", err
	}
	return info, file, nil
}

// GoMod is like Lookup(ctx, path).GoMod(rev) but avoids the
// repository path resolution in Lookup if the result is
// already cached on local disk.
func (f *Fetcher) GoMod(ctx context.Context, path, rev string) ([]byte, error) {
	// Convert commit hash to pseudo-version
	// to increase cache hit rate.
	if !gover.ModIsValid(path, rev) {
		if _, info, err := readDiskStat(ctx, path, rev); err == nil {
			rev = info.Version
		} else {
			if errors.Is(err, statCacheErr) {
				return nil, err
			}
			err := TryProxies(func(proxy string) error {
				info, err := f.Lookup(ctx, proxy, path).Stat(ctx, rev)
				if err == nil {
					rev = info.Version
				}
				return err
			})
			if err != nil {
				return nil, err
			}
		}
	}

	_, data, err := f.readDiskGoMod(ctx, path, rev)
	if err == nil {
		return data, nil
	}

	err = TryProxies(func(proxy string) (err error) {
		data, err = f.Lookup(ctx, proxy, path).GoMod(ctx, rev)
		return err
	})
	return data, err
}

// GoModFile is like GoMod but returns the name of the file containing
// the cached information.
func (f *Fetcher) GoModFile(ctx context.Context, path, version string) (string, error) {
	if !gover.ModIsValid(path, version) {
		return "", fmt.Errorf("invalid version %q", version)
	}
	if _, err := f.GoMod(ctx, path, version); err != nil {
		return "", err
	}
	// GoMod should have populated the disk cache for us.
	file, err := CachePath(ctx, module.Version{Path: path, Version: version}, "mod")
	if err != nil {
		return "", err
	}
	return file, nil
}

// GoModSum returns the go.sum entry for the module version's go.mod file.
// (That is, it returns the entry listed in go.sum as "path version/go.mod".)
func (f *Fetcher) GoModSum(ctx context.Context, path, version string) (string, error) {
	if !gover.ModIsValid(path, version) {
		return "", fmt.Errorf("invalid version %q", version)
	}
	data, err := f.GoMod(ctx, path, version)
	if err != nil {
		return "", err
	}
	sum, err := goModSum(data)
	if err != nil {
		return "", err
	}
	return sum, nil
}

var errNotCached = fmt.Errorf("not in cache")

// readDiskStat reads a cached stat result from disk,
// returning the name of the cache file and the result.
// If the read fails, the caller can use
// writeDiskStat(file, info) to write a new cache entry.
func readDiskStat(ctx context.Context, path, rev string) (file string, info *RevInfo, err error) {
	if gover.IsToolchain(path) {
		return "", nil, errNotCached
	}
	file, data, err := readDiskCache(ctx, path, rev, "info")
	if err != nil {
		// If the cache already contains a pseudo-version with the given hash, we
		// would previously return that pseudo-version without checking upstream.
		// However, that produced an unfortunate side-effect: if the author added a
		// tag to the repository, 'go get' would not pick up the effect of that new
		// tag on the existing commits, and 'go' commands that referred to those
		// commits would use the previous name instead of the new one.
		//
		// That's especially problematic if the original pseudo-version starts with
		// v0.0.0-, as was the case for all pseudo-versions during vgo development,
		// since a v0.0.0- pseudo-version has lower precedence than pretty much any
		// tagged version.
		//
		// In practice, we're only looking up by hash during initial conversion of a
		// legacy config and during an explicit 'go get', and a little extra latency
		// for those operations seems worth the benefit of picking up more accurate
		// versions.
		//
		// Fall back to this resolution scheme only if the GOPROXY setting prohibits
		// us from resolving upstream tags.
		if cfg.GOPROXY == "off" {
			if file, info, err := readDiskStatByHash(ctx, path, rev); err == nil {
				return file, info, nil
			}
		}
		return file, nil, err
	}
	info = new(RevInfo)
	if err := json.Unmarshal(data, info); err != nil {
		return file, nil, errNotCached
	}
	// The disk might have stale .info files that have Name and Short fields set.
	// We want to canonicalize to .info files with those fields omitted.
	// Remarshal and update the cache file if needed.
	data2, err := json.Marshal(info)
	if err == nil && !bytes.Equal(data2, data) {
		writeDiskCache(ctx, file, data)
	}
	return file, info, nil
}

// readDiskStatByHash is a fallback for readDiskStat for the case
// where rev is a commit hash instead of a proper semantic version.
// In that case, we look for a cached pseudo-version that matches
// the commit hash. If we find one, we use it.
// This matters most for converting legacy package management
// configs, when we are often looking up commits by full hash.
// Without this check we'd be doing network I/O to the remote repo
// just to find out about a commit we already know about
// (and have cached under its pseudo-version).
func readDiskStatByHash(ctx context.Context, path, rev string) (file string, info *RevInfo, err error) {
	if gover.IsToolchain(path) {
		return "", nil, errNotCached
	}
	if cfg.GOMODCACHE == "" {
		// Do not download to current directory.
		return "", nil, errNotCached
	}

	if !codehost.AllHex(rev) || len(rev) < 12 {
		return "", nil, errNotCached
	}
	rev = rev[:12]
	cdir, err := cacheDir(ctx, path)
	if err != nil {
		return "", nil, errNotCached
	}
	dir, err := os.Open(cdir)
	if err != nil {
		return "", nil, errNotCached
	}
	names, err := dir.Readdirnames(-1)
	dir.Close()
	if err != nil {
		return "", nil, errNotCached
	}

	// A given commit hash may map to more than one pseudo-version,
	// depending on which tags are present on the repository.
	// Take the highest such version.
	var maxVersion string
	suffix := "-" + rev + ".info"
	err = errNotCached
	for _, name := range names {
		if strings.HasSuffix(name, suffix) {
			v := strings.TrimSuffix(name, ".info")
			if module.IsPseudoVersion(v) && semver.Compare(v, maxVersion) > 0 {
				maxVersion = v
				file, info, err = readDiskStat(ctx, path, strings.TrimSuffix(name, ".info"))
			}
		}
	}
	return file, info, err
}

// oldVgoPrefix is the prefix in the old auto-generated cached go.mod files.
// We stopped trying to auto-generate the go.mod files. Now we use a trivial
// go.mod with only a module line, and we've dropped the version prefix
// entirely. If we see a version prefix, that means we're looking at an old copy
// and should ignore it.
var oldVgoPrefix = []byte("//vgo 0.0.")

// readDiskGoMod reads a cached go.mod file from disk,
// returning the name of the cache file and the result.
// If the read fails, the caller can use
// writeDiskGoMod(file, data) to write a new cache entry.
func (f *Fetcher) readDiskGoMod(ctx context.Context, path, rev string) (file string, data []byte, err error) {
	if gover.IsToolchain(path) {
		return "", nil, errNotCached
	}
	file, data, err = readDiskCache(ctx, path, rev, "mod")

	// If the file has an old auto-conversion prefix, pretend it's not there.
	if bytes.HasPrefix(data, oldVgoPrefix) {
		err = errNotCached
		data = nil
	}

	if err == nil {
		if err := checkGoMod(f, path, rev, data); err != nil {
			return "", nil, err
		}
	}

	return file, data, err
}

// readDiskCache is the generic "read from a cache file" implementation.
// It takes the revision and an identifying suffix for the kind of data being cached.
// It returns the name of the cache file and the content of the file.
// If the read fails, the caller can use
// writeDiskCache(file, data) to write a new cache entry.
func readDiskCache(ctx context.Context, path, rev, suffix string) (file string, data []byte, err error) {
	if gover.IsToolchain(path) {
		return "", nil, errNotCached
	}
	file, err = CachePath(ctx, module.Version{Path: path, Version: rev}, suffix)
	if err != nil {
		return "", nil, errNotCached
	}
	data, err = robustio.ReadFile(file)
	if err != nil {
		return file, nil, errNotCached
	}
	return file, data, nil
}

// writeDiskStat writes a stat result cache entry.
// The file name must have been returned by a previous call to readDiskStat.
func writeDiskStat(ctx context.Context, file string, info *RevInfo) error {
	if file == "" {
		return nil
	}

	if info.Origin != nil {
		// Clean the origin information, which might have too many
		// validation criteria, for example if we are saving the result of
		// m@master as m@pseudo-version.
		clean := *info
		info = &clean
		o := *info.Origin
		info.Origin = &o

		// Tags and RepoSum never matter if you are starting with a semver version,
		// as we would be when finding this cache entry.
		o.TagSum = ""
		o.TagPrefix = ""
		o.RepoSum = ""
		// Ref doesn't matter if you have a pseudoversion.
		if module.IsPseudoVersion(info.Version) {
			o.Ref = ""
		}
	}

	js, err := json.Marshal(info)
	if err != nil {
		return err
	}
	return writeDiskCache(ctx, file, js)
}

// writeDiskGoMod writes a go.mod cache entry.
// The file name must have been returned by a previous call to readDiskGoMod.
func writeDiskGoMod(ctx context.Context, file string, text []byte) error {
	return writeDiskCache(ctx, file, text)
}

// writeDiskCache is the generic "write to a cache file" implementation.
// The file must have been returned by a previous call to readDiskCache.
func writeDiskCache(ctx context.Context, file string, data []byte) error {
	if file == "" {
		return nil
	}
	// Make sure directory for file exists.
	if err := os.MkdirAll(filepath.Dir(file), 0o777); err != nil {
		return err
	}

	// Write the file to a temporary location, and then rename it to its final
	// path to reduce the likelihood of a corrupt file existing at that final path.
	f, err := tempFile(ctx, filepath.Dir(file), filepath.Base(file), 0o666)
	if err != nil {
		return err
	}
	defer func() {
		// Only call os.Remove on f.Name() if we failed to rename it: otherwise,
		// some other process may have created a new file with the same name after
		// the rename completed.
		if err != nil {
			f.Close()
			os.Remove(f.Name())
		}
	}()

	if _, err := f.Write(data); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := robustio.Rename(f.Name(), file); err != nil {
		return err
	}

	if strings.HasSuffix(file, ".mod") {
		rewriteVersionList(ctx, filepath.Dir(file))
	}
	return nil
}

// tempFile creates a new temporary file with given permission bits.
func tempFile(ctx context.Context, dir, prefix string, perm fs.FileMode) (f *os.File, err error) {
	for i := 0; i < 10000; i++ {
		name := filepath.Join(dir, prefix+strconv.Itoa(rand.Intn(1000000000))+".tmp")
		f, err = os.OpenFile(name, os.O_RDWR|os.O_CREATE|os.O_EXCL, perm)
		if os.IsExist(err) {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			continue
		}
		break
	}
	return
}

// rewriteVersionList rewrites the version list in dir
// after a new *.mod file has been written.
func rewriteVersionList(ctx context.Context, dir string) (err error) {
	if filepath.Base(dir) != "@v" {
		base.Fatalf("go: internal error: misuse of rewriteVersionList")
	}

	listFile := filepath.Join(dir, "list")

	// Lock listfile when writing to it to try to avoid corruption to the file.
	// Under rare circumstances, for instance, if the system loses power in the
	// middle of a write it is possible for corrupt data to be written. This is
	// not a problem for the go command itself, but may be an issue if the
	// cache is being served by a GOPROXY HTTP server. This will be corrected
	// the next time a new version of the module is fetched and the file is rewritten.
	// TODO(matloob): golang.org/issue/43313 covers adding a go mod verify
	// command that removes module versions that fail checksums. It should also
	// remove list files that are detected to be corrupt.
	f, err := lockedfile.Edit(listFile)
	if err != nil {
		return err
	}
	defer func() {
		if cerr := f.Close(); cerr != nil && err == nil {
			err = cerr
		}
	}()
	infos, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	var list []string
	for _, info := range infos {
		// We look for *.mod files on the theory that if we can't supply
		// the .mod file then there's no point in listing that version,
		// since it's unusable. (We can have *.info without *.mod.)
		// We don't require *.zip files on the theory that for code only
		// involved in module graph construction, many *.zip files
		// will never be requested.
		name := info.Name()
		if v, found := strings.CutSuffix(name, ".mod"); found {
			if v != "" && module.CanonicalVersion(v) == v {
				list = append(list, v)
			}
		}
	}
	semver.Sort(list)

	var buf bytes.Buffer
	for _, v := range list {
		buf.WriteString(v)
		buf.WriteString("\n")
	}
	if fi, err := f.Stat(); err == nil && int(fi.Size()) == buf.Len() {
		old := make([]byte, buf.Len()+1)
		if n, err := f.ReadAt(old, 0); err == io.EOF && n == buf.Len() && bytes.Equal(buf.Bytes(), old) {
			return nil // No edit needed.
		}
	}
	// Remove existing contents, so that when we truncate to the actual size it will zero-fill,
	// and we will be able to detect (some) incomplete writes as files containing trailing NUL bytes.
	if err := f.Truncate(0); err != nil {
		return err
	}
	// Reserve the final size and zero-fill.
	if err := f.Truncate(int64(buf.Len())); err != nil {
		return err
	}
	// Write the actual contents. If this fails partway through,
	// the remainder of the file should remain as zeroes.
	if _, err := f.Write(buf.Bytes()); err != nil {
		f.Truncate(0)
		return err
	}

	return nil
}

var (
	statCacheOnce sync.Once
	statCacheErr  error

	counterErrorsGOMODCACHEEntryRelative = counter.New("go/errors:gomodcache-entry-relative")
)

// checkCacheDir checks if the directory specified by GOMODCACHE exists. An
// error is returned if it does not.
func checkCacheDir(ctx context.Context) error {
	if cfg.GOMODCACHE == "" {
		// modload.Init exits if GOPATH[0] is empty, and cfg.GOMODCACHE
		// is set to GOPATH[0]/pkg/mod if GOMODCACHE is empty, so this should never happen.
		return fmt.Errorf("module cache not found: neither GOMODCACHE nor GOPATH is set")
	}
	if !filepath.IsAbs(cfg.GOMODCACHE) {
		counterErrorsGOMODCACHEEntryRelative.Inc()
		return fmt.Errorf("GOMODCACHE entry is relative; must be absolute path: %q.\n", cfg.GOMODCACHE)
	}

	// os.Stat is slow on Windows, so we only call it once to prevent unnecessary
	// I/O every time this function is called.
	statCacheOnce.Do(func() {
		fi, err := os.Stat(cfg.GOMODCACHE)
		if err != nil {
			if !os.IsNotExist(err) {
				statCacheErr = fmt.Errorf("could not create module cache: %w", err)
				return
			}
			if err := os.MkdirAll(cfg.GOMODCACHE, 0o777); err != nil {
				statCacheErr = fmt.Errorf("could not create module cache: %w", err)
				return
			}
			return
		}
		if !fi.IsDir() {
			statCacheErr = fmt.Errorf("could not create module cache: %q is not a directory", cfg.GOMODCACHE)
			return
		}
	})
	return statCacheErr
}
