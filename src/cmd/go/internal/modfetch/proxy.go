// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	url "net/url"
	"os"
	pathpkg "path"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
	"unicode"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/module"
	"cmd/go/internal/semver"
	"cmd/go/internal/web"
)

var HelpGoproxy = &base.Command{
	UsageLine: "goproxy",
	Short:     "module proxy protocol",
	Long: `
The go command by default downloads modules from version control systems
directly, just as 'go get' always has. The GOPROXY environment variable allows
further control over the download source. If GOPROXY is unset, is the empty string,
or is the string "direct", downloads use the default direct connection to version
control systems. Setting GOPROXY to "off" disallows downloading modules from
any source. Otherwise, GOPROXY is expected to be a comma-separated list of
the URLs of module proxies, in which case the go command will fetch modules
from those proxies. For each request, the go command tries each proxy in sequence,
only moving to the next if the current proxy returns a 404 or 410 HTTP response.
The string "direct" may appear in the proxy list, to cause a direct connection to
be attempted at that point in the search.

No matter the source of the modules, downloaded modules must match existing
entries in go.sum (see 'go help modules' for discussion of verification).

A Go module proxy is any web server that can respond to GET requests for
URLs of a specified form. The requests have no query parameters, so even
a site serving from a fixed file system (including a file:/// URL)
can be a module proxy.

The GET requests sent to a Go module proxy are:

GET $GOPROXY/<module>/@v/list returns a list of all known versions of the
given module, one per line.

GET $GOPROXY/<module>/@v/<version>.info returns JSON-formatted metadata
about that version of the given module.

GET $GOPROXY/<module>/@v/<version>.mod returns the go.mod file
for that version of the given module.

GET $GOPROXY/<module>/@v/<version>.zip returns the zip archive
for that version of the given module.

To avoid problems when serving from case-sensitive file systems,
the <module> and <version> elements are case-encoded, replacing every
uppercase letter with an exclamation mark followed by the corresponding
lower-case letter: github.com/Azure encodes as github.com/!azure.

The JSON-formatted metadata about a given module corresponds to
this Go data structure, which may be expanded in the future:

    type Info struct {
        Version string    // version string
        Time    time.Time // commit time
    }

The zip archive for a specific version of a given module is a
standard zip file that contains the file tree corresponding
to the module's source code and related files. The archive uses
slash-separated paths, and every file path in the archive must
begin with <module>@<version>/, where the module and version are
substituted directly, not case-encoded. The root of the module
file tree corresponds to the <module>@<version>/ prefix in the
archive.

Even when downloading directly from version control systems,
the go command synthesizes explicit info, mod, and zip files
and stores them in its local cache, $GOPATH/pkg/mod/cache/download,
the same as if it had downloaded them directly from a proxy.
The cache layout is the same as the proxy URL space, so
serving $GOPATH/pkg/mod/cache/download at (or copying it to)
https://example.com/proxy would let other users access those
cached module versions with GOPROXY=https://example.com/proxy.
`,
}

var proxyURL = cfg.Getenv("GOPROXY")

// SetProxy sets the proxy to use when fetching modules.
// It accepts the same syntax as the GOPROXY environment variable,
// which also provides its default configuration.
// SetProxy must not be called after the first module fetch has begun.
func SetProxy(url string) {
	proxyURL = url
}

var proxyOnce struct {
	sync.Once
	list []string
	err  error
}

func proxyURLs() ([]string, error) {
	proxyOnce.Do(func() {
		for _, proxyURL := range strings.Split(proxyURL, ",") {
			if proxyURL == "" {
				continue
			}
			if proxyURL == "direct" {
				proxyOnce.list = append(proxyOnce.list, "direct")
				continue
			}

			// Check that newProxyRepo accepts the URL.
			// It won't do anything with the path.
			_, err := newProxyRepo(proxyURL, "golang.org/x/text")
			if err != nil {
				proxyOnce.err = err
				return
			}
			proxyOnce.list = append(proxyOnce.list, proxyURL)
		}
	})

	return proxyOnce.list, proxyOnce.err
}

func lookupProxy(path string) (Repo, error) {
	list, err := proxyURLs()
	if err != nil {
		return nil, err
	}

	var repos listRepo
	for _, u := range list {
		var r Repo
		if u == "direct" {
			// lookupDirect does actual network traffic.
			// Especially if GOPROXY="http://mainproxy,direct",
			// avoid the network until we need it by using a lazyRepo wrapper.
			r = &lazyRepo{setup: lookupDirect, path: path}
		} else {
			// The URL itself was checked in proxyURLs.
			// The only possible error here is a bad path,
			// so we can return it unconditionally.
			r, err = newProxyRepo(u, path)
			if err != nil {
				return nil, err
			}
		}
		repos = append(repos, r)
	}
	return repos, nil
}

type proxyRepo struct {
	url  *url.URL
	path string
}

func newProxyRepo(baseURL, path string) (Repo, error) {
	base, err := url.Parse(baseURL)
	if err != nil {
		return nil, err
	}
	switch base.Scheme {
	case "http", "https":
		// ok
	case "file":
		if *base != (url.URL{Scheme: base.Scheme, Path: base.Path, RawPath: base.RawPath}) {
			return nil, fmt.Errorf("invalid file:// proxy URL with non-path elements: %s", web.Redacted(base))
		}
	case "":
		return nil, fmt.Errorf("invalid proxy URL missing scheme: %s", web.Redacted(base))
	default:
		return nil, fmt.Errorf("invalid proxy URL scheme (must be https, http, file): %s", web.Redacted(base))
	}

	enc, err := module.EncodePath(path)
	if err != nil {
		return nil, err
	}

	base.Path = strings.TrimSuffix(base.Path, "/") + "/" + enc
	base.RawPath = strings.TrimSuffix(base.RawPath, "/") + "/" + pathEscape(enc)
	return &proxyRepo{base, path}, nil
}

func (p *proxyRepo) ModulePath() string {
	return p.path
}

func (p *proxyRepo) getBytes(path string) ([]byte, error) {
	body, err := p.getBody(path)
	if err != nil {
		return nil, err
	}
	defer body.Close()
	return ioutil.ReadAll(body)
}

func (p *proxyRepo) getBody(path string) (io.ReadCloser, error) {
	fullPath := pathpkg.Join(p.url.Path, path)
	if p.url.Scheme == "file" {
		rawPath, err := url.PathUnescape(fullPath)
		if err != nil {
			return nil, err
		}
		if runtime.GOOS == "windows" && len(rawPath) >= 4 && rawPath[0] == '/' && unicode.IsLetter(rune(rawPath[1])) && rawPath[2] == ':' {
			// On Windows, file URLs look like "file:///C:/foo/bar". url.Path will
			// start with a slash which must be removed. See golang.org/issue/6027.
			rawPath = rawPath[1:]
		}
		return os.Open(filepath.FromSlash(rawPath))
	}

	target := *p.url
	target.Path = fullPath
	target.RawPath = pathpkg.Join(target.RawPath, pathEscape(path))

	resp, err := web.Get(web.DefaultSecurity, &target)
	if err != nil {
		return nil, err
	}
	if err := resp.Err(); err != nil {
		resp.Body.Close()
		return nil, err
	}
	return resp.Body, nil
}

func (p *proxyRepo) Versions(prefix string) ([]string, error) {
	data, err := p.getBytes("@v/list")
	if err != nil {
		return nil, err
	}
	var list []string
	for _, line := range strings.Split(string(data), "\n") {
		f := strings.Fields(line)
		if len(f) >= 1 && semver.IsValid(f[0]) && strings.HasPrefix(f[0], prefix) {
			list = append(list, f[0])
		}
	}
	SortVersions(list)
	return list, nil
}

func (p *proxyRepo) latest() (*RevInfo, error) {
	data, err := p.getBytes("@v/list")
	if err != nil {
		return nil, err
	}
	var best time.Time
	var bestVersion string
	for _, line := range strings.Split(string(data), "\n") {
		f := strings.Fields(line)
		if len(f) >= 2 && semver.IsValid(f[0]) {
			ft, err := time.Parse(time.RFC3339, f[1])
			if err == nil && best.Before(ft) {
				best = ft
				bestVersion = f[0]
			}
		}
	}
	if bestVersion == "" {
		return nil, fmt.Errorf("no commits")
	}
	info := &RevInfo{
		Version: bestVersion,
		Name:    bestVersion,
		Short:   bestVersion,
		Time:    best,
	}
	return info, nil
}

func (p *proxyRepo) Stat(rev string) (*RevInfo, error) {
	encRev, err := module.EncodeVersion(rev)
	if err != nil {
		return nil, err
	}
	data, err := p.getBytes("@v/" + encRev + ".info")
	if err != nil {
		return nil, err
	}
	info := new(RevInfo)
	if err := json.Unmarshal(data, info); err != nil {
		return nil, err
	}
	return info, nil
}

func (p *proxyRepo) Latest() (*RevInfo, error) {
	data, err := p.getBytes("@latest")
	if err != nil {
		// TODO return err if not 404
		return p.latest()
	}
	info := new(RevInfo)
	if err := json.Unmarshal(data, info); err != nil {
		return nil, err
	}
	return info, nil
}

func (p *proxyRepo) GoMod(version string) ([]byte, error) {
	encVer, err := module.EncodeVersion(version)
	if err != nil {
		return nil, err
	}
	data, err := p.getBytes("@v/" + encVer + ".mod")
	if err != nil {
		return nil, err
	}
	return data, nil
}

func (p *proxyRepo) Zip(dst io.Writer, version string) error {
	encVer, err := module.EncodeVersion(version)
	if err != nil {
		return err
	}
	body, err := p.getBody("@v/" + encVer + ".zip")
	if err != nil {
		return err
	}
	defer body.Close()

	lr := &io.LimitedReader{R: body, N: codehost.MaxZipFile + 1}
	if _, err := io.Copy(dst, lr); err != nil {
		return err
	}
	if lr.N <= 0 {
		return fmt.Errorf("downloaded zip file too large")
	}
	return nil
}

// pathEscape escapes s so it can be used in a path.
// That is, it escapes things like ? and # (which really shouldn't appear anyway).
// It does not escape / to %2F: our REST API is designed so that / can be left as is.
func pathEscape(s string) string {
	return strings.ReplaceAll(url.PathEscape(s), "%2F", "/")
}

// A lazyRepo is a lazily-initialized Repo,
// constructed on demand by calling setup.
type lazyRepo struct {
	path  string
	setup func(string) (Repo, error)
	once  sync.Once
	repo  Repo
	err   error
}

func (r *lazyRepo) init() {
	r.repo, r.err = r.setup(r.path)
}

func (r *lazyRepo) ModulePath() string {
	return r.path
}

func (r *lazyRepo) Versions(prefix string) ([]string, error) {
	if r.once.Do(r.init); r.err != nil {
		return nil, r.err
	}
	return r.repo.Versions(prefix)
}

func (r *lazyRepo) Stat(rev string) (*RevInfo, error) {
	if r.once.Do(r.init); r.err != nil {
		return nil, r.err
	}
	return r.repo.Stat(rev)
}

func (r *lazyRepo) Latest() (*RevInfo, error) {
	if r.once.Do(r.init); r.err != nil {
		return nil, r.err
	}
	return r.repo.Latest()
}

func (r *lazyRepo) GoMod(version string) ([]byte, error) {
	if r.once.Do(r.init); r.err != nil {
		return nil, r.err
	}
	return r.repo.GoMod(version)
}

func (r *lazyRepo) Zip(dst io.Writer, version string) error {
	if r.once.Do(r.init); r.err != nil {
		return r.err
	}
	return r.repo.Zip(dst, version)
}

// A listRepo is a preference list of Repos.
// The list must be non-empty and all Repos
// must return the same result from ModulePath.
// For each method, the repos are tried in order
// until one succeeds or returns a non-ErrNotExist (non-404) error.
type listRepo []Repo

func (l listRepo) ModulePath() string {
	return l[0].ModulePath()
}

func (l listRepo) Versions(prefix string) ([]string, error) {
	for i, r := range l {
		v, err := r.Versions(prefix)
		if i == len(l)-1 || !errors.Is(err, os.ErrNotExist) {
			return v, err
		}
	}
	panic("no repos")
}

func (l listRepo) Stat(rev string) (*RevInfo, error) {
	for i, r := range l {
		info, err := r.Stat(rev)
		if i == len(l)-1 || !errors.Is(err, os.ErrNotExist) {
			return info, err
		}
	}
	panic("no repos")
}

func (l listRepo) Latest() (*RevInfo, error) {
	for i, r := range l {
		info, err := r.Latest()
		if i == len(l)-1 || !errors.Is(err, os.ErrNotExist) {
			return info, err
		}
	}
	panic("no repos")
}

func (l listRepo) GoMod(version string) ([]byte, error) {
	for i, r := range l {
		data, err := r.GoMod(version)
		if i == len(l)-1 || !errors.Is(err, os.ErrNotExist) {
			return data, err
		}
	}
	panic("no repos")
}

func (l listRepo) Zip(dst io.Writer, version string) error {
	for i, r := range l {
		err := r.Zip(dst, version)
		if i == len(l)-1 || !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	panic("no repos")
}
