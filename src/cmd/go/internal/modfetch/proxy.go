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
	"net/url"
	"os"
	"path"
	pathpkg "path"
	"path/filepath"
	"strings"
	"sync"
	"time"

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

var proxyOnce struct {
	sync.Once
	list []string
	err  error
}

func proxyURLs() ([]string, error) {
	proxyOnce.Do(func() {
		if cfg.GONOPROXY != "" && cfg.GOPROXY != "direct" {
			proxyOnce.list = append(proxyOnce.list, "noproxy")
		}
		for _, proxyURL := range strings.Split(cfg.GOPROXY, ",") {
			proxyURL = strings.TrimSpace(proxyURL)
			if proxyURL == "" {
				continue
			}
			if proxyURL == "off" {
				// "off" always fails hard, so can stop walking list.
				proxyOnce.list = append(proxyOnce.list, "off")
				break
			}
			if proxyURL == "direct" {
				proxyOnce.list = append(proxyOnce.list, "direct")
				// For now, "direct" is the end of the line. We may decide to add some
				// sort of fallback behavior for them in the future, so ignore
				// subsequent entries for forward-compatibility.
				break
			}

			// Single-word tokens are reserved for built-in behaviors, and anything
			// containing the string ":/" or matching an absolute file path must be a
			// complete URL. For all other paths, implicitly add "https://".
			if strings.ContainsAny(proxyURL, ".:/") && !strings.Contains(proxyURL, ":/") && !filepath.IsAbs(proxyURL) && !path.IsAbs(proxyURL) {
				proxyURL = "https://" + proxyURL
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

// TryProxies iterates f over each configured proxy (including "noproxy" and
// "direct" if applicable) until f returns an error that is not
// equivalent to os.ErrNotExist.
//
// TryProxies then returns that final error.
//
// If GOPROXY is set to "off", TryProxies invokes f once with the argument
// "off".
func TryProxies(f func(proxy string) error) error {
	proxies, err := proxyURLs()
	if err != nil {
		return err
	}
	if len(proxies) == 0 {
		return f("off")
	}

	for _, proxy := range proxies {
		err = f(proxy)
		if !errors.Is(err, os.ErrNotExist) {
			break
		}
	}
	return err
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

// versionError returns err wrapped in a ModuleError for p.path.
func (p *proxyRepo) versionError(version string, err error) error {
	if version != "" && version != module.CanonicalVersion(version) {
		return &module.ModuleError{
			Path: p.path,
			Err: &module.InvalidVersionError{
				Version: version,
				Pseudo:  IsPseudoVersion(version),
				Err:     err,
			},
		}
	}

	return &module.ModuleError{
		Path:    p.path,
		Version: version,
		Err:     err,
	}
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
		return nil, p.versionError("", err)
	}
	var list []string
	for _, line := range strings.Split(string(data), "\n") {
		f := strings.Fields(line)
		if len(f) >= 1 && semver.IsValid(f[0]) && strings.HasPrefix(f[0], prefix) && !IsPseudoVersion(f[0]) {
			list = append(list, f[0])
		}
	}
	SortVersions(list)
	return list, nil
}

func (p *proxyRepo) latest() (*RevInfo, error) {
	data, err := p.getBytes("@v/list")
	if err != nil {
		return nil, p.versionError("", err)
	}

	var (
		bestTime             time.Time
		bestTimeIsFromPseudo bool
		bestVersion          string
	)

	for _, line := range strings.Split(string(data), "\n") {
		f := strings.Fields(line)
		if len(f) >= 1 && semver.IsValid(f[0]) {
			// If the proxy includes timestamps, prefer the timestamp it reports.
			// Otherwise, derive the timestamp from the pseudo-version.
			var (
				ft             time.Time
				ftIsFromPseudo = false
			)
			if len(f) >= 2 {
				ft, _ = time.Parse(time.RFC3339, f[1])
			} else if IsPseudoVersion(f[0]) {
				ft, _ = PseudoVersionTime(f[0])
				ftIsFromPseudo = true
			} else {
				// Repo.Latest promises that this method is only called where there are
				// no tagged versions. Ignore any tagged versions that were added in the
				// meantime.
				continue
			}
			if bestTime.Before(ft) {
				bestTime = ft
				bestTimeIsFromPseudo = ftIsFromPseudo
				bestVersion = f[0]
			}
		}
	}
	if bestVersion == "" {
		return nil, p.versionError("", codehost.ErrNoCommits)
	}

	if bestTimeIsFromPseudo {
		// We parsed bestTime from the pseudo-version, but that's in UTC and we're
		// supposed to report the timestamp as reported by the VCS.
		// Stat the selected version to canonicalize the timestamp.
		//
		// TODO(bcmills): Should we also stat other versions to ensure that we
		// report the correct Name and Short for the revision?
		return p.Stat(bestVersion)
	}

	return &RevInfo{
		Version: bestVersion,
		Name:    bestVersion,
		Short:   bestVersion,
		Time:    bestTime,
	}, nil
}

func (p *proxyRepo) Stat(rev string) (*RevInfo, error) {
	encRev, err := module.EncodeVersion(rev)
	if err != nil {
		return nil, p.versionError(rev, err)
	}
	data, err := p.getBytes("@v/" + encRev + ".info")
	if err != nil {
		return nil, p.versionError(rev, err)
	}
	info := new(RevInfo)
	if err := json.Unmarshal(data, info); err != nil {
		return nil, p.versionError(rev, err)
	}
	if info.Version != rev && rev == module.CanonicalVersion(rev) && module.Check(p.path, rev) == nil {
		// If we request a correct, appropriate version for the module path, the
		// proxy must return either exactly that version or an error — not some
		// arbitrary other version.
		return nil, p.versionError(rev, fmt.Errorf("proxy returned info for version %s instead of requested version", info.Version))
	}
	return info, nil
}

func (p *proxyRepo) Latest() (*RevInfo, error) {
	data, err := p.getBytes("@latest")
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			return nil, p.versionError("", err)
		}
		return p.latest()
	}
	info := new(RevInfo)
	if err := json.Unmarshal(data, info); err != nil {
		return nil, p.versionError("", err)
	}
	return info, nil
}

func (p *proxyRepo) GoMod(version string) ([]byte, error) {
	if version != module.CanonicalVersion(version) {
		return nil, p.versionError(version, fmt.Errorf("internal error: version passed to GoMod is not canonical"))
	}

	encVer, err := module.EncodeVersion(version)
	if err != nil {
		return nil, p.versionError(version, err)
	}
	data, err := p.getBytes("@v/" + encVer + ".mod")
	if err != nil {
		return nil, p.versionError(version, err)
	}
	return data, nil
}

func (p *proxyRepo) Zip(dst io.Writer, version string) error {
	if version != module.CanonicalVersion(version) {
		return p.versionError(version, fmt.Errorf("internal error: version passed to Zip is not canonical"))
	}

	encVer, err := module.EncodeVersion(version)
	if err != nil {
		return p.versionError(version, err)
	}
	body, err := p.getBody("@v/" + encVer + ".zip")
	if err != nil {
		return p.versionError(version, err)
	}
	defer body.Close()

	lr := &io.LimitedReader{R: body, N: codehost.MaxZipFile + 1}
	if _, err := io.Copy(dst, lr); err != nil {
		return p.versionError(version, err)
	}
	if lr.N <= 0 {
		return p.versionError(version, fmt.Errorf("downloaded zip file too large"))
	}
	return nil
}

// pathEscape escapes s so it can be used in a path.
// That is, it escapes things like ? and # (which really shouldn't appear anyway).
// It does not escape / to %2F: our REST API is designed so that / can be left as is.
func pathEscape(s string) string {
	return strings.ReplaceAll(url.PathEscape(s), "%2F", "/")
}
