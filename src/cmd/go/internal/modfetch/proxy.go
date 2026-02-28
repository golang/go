// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/url"
	pathpkg "path"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/web"

	"golang.org/x/mod/module"
	"golang.org/x/mod/semver"
)

var HelpGoproxy = &base.Command{
	UsageLine: "goproxy",
	Short:     "module proxy protocol",
	Long: `
A Go module proxy is any web server that can respond to GET requests for
URLs of a specified form. The requests have no query parameters, so even
a site serving from a fixed file system (including a file:/// URL)
can be a module proxy.

For details on the GOPROXY protocol, see
https://golang.org/ref/mod#goproxy-protocol.
`,
}

var proxyOnce struct {
	sync.Once
	list []proxySpec
	err  error
}

type proxySpec struct {
	// url is the proxy URL or one of "off", "direct", "noproxy".
	url string

	// fallBackOnError is true if a request should be attempted on the next proxy
	// in the list after any error from this proxy. If fallBackOnError is false,
	// the request will only be attempted on the next proxy if the error is
	// equivalent to os.ErrNotFound, which is true for 404 and 410 responses.
	fallBackOnError bool
}

func proxyList() ([]proxySpec, error) {
	proxyOnce.Do(func() {
		if cfg.GONOPROXY != "" && cfg.GOPROXY != "direct" {
			proxyOnce.list = append(proxyOnce.list, proxySpec{url: "noproxy"})
		}

		goproxy := cfg.GOPROXY
		for goproxy != "" {
			var url string
			fallBackOnError := false
			if i := strings.IndexAny(goproxy, ",|"); i >= 0 {
				url = goproxy[:i]
				fallBackOnError = goproxy[i] == '|'
				goproxy = goproxy[i+1:]
			} else {
				url = goproxy
				goproxy = ""
			}

			url = strings.TrimSpace(url)
			if url == "" {
				continue
			}
			if url == "off" {
				// "off" always fails hard, so can stop walking list.
				proxyOnce.list = append(proxyOnce.list, proxySpec{url: "off"})
				break
			}
			if url == "direct" {
				proxyOnce.list = append(proxyOnce.list, proxySpec{url: "direct"})
				// For now, "direct" is the end of the line. We may decide to add some
				// sort of fallback behavior for them in the future, so ignore
				// subsequent entries for forward-compatibility.
				break
			}

			// Single-word tokens are reserved for built-in behaviors, and anything
			// containing the string ":/" or matching an absolute file path must be a
			// complete URL. For all other paths, implicitly add "https://".
			if strings.ContainsAny(url, ".:/") && !strings.Contains(url, ":/") && !filepath.IsAbs(url) && !pathpkg.IsAbs(url) {
				url = "https://" + url
			}

			// Check that newProxyRepo accepts the URL.
			// It won't do anything with the path.
			if _, err := newProxyRepo(url, "golang.org/x/text"); err != nil {
				proxyOnce.err = err
				return
			}

			proxyOnce.list = append(proxyOnce.list, proxySpec{
				url:             url,
				fallBackOnError: fallBackOnError,
			})
		}

		if len(proxyOnce.list) == 0 ||
			len(proxyOnce.list) == 1 && proxyOnce.list[0].url == "noproxy" {
			// There were no proxies, other than the implicit "noproxy" added when
			// GONOPROXY is set. This can happen if GOPROXY is a non-empty string
			// like "," or " ".
			proxyOnce.err = fmt.Errorf("GOPROXY list is not the empty string, but contains no entries")
		}
	})

	return proxyOnce.list, proxyOnce.err
}

// TryProxies iterates f over each configured proxy (including "noproxy" and
// "direct" if applicable) until f returns no error or until f returns an
// error that is not equivalent to fs.ErrNotExist on a proxy configured
// not to fall back on errors.
//
// TryProxies then returns that final error.
//
// If GOPROXY is set to "off", TryProxies invokes f once with the argument
// "off".
func TryProxies(f func(proxy string) error) error {
	proxies, err := proxyList()
	if err != nil {
		return err
	}
	if len(proxies) == 0 {
		panic("GOPROXY list is empty")
	}

	// We try to report the most helpful error to the user. "direct" and "noproxy"
	// errors are best, followed by proxy errors other than ErrNotExist, followed
	// by ErrNotExist.
	//
	// Note that errProxyOff, errNoproxy, and errUseProxy are equivalent to
	// ErrNotExist. errUseProxy should only be returned if "noproxy" is the only
	// proxy. errNoproxy should never be returned, since there should always be a
	// more useful error from "noproxy" first.
	const (
		notExistRank = iota
		proxyRank
		directRank
	)
	var bestErr error
	bestErrRank := notExistRank
	for _, proxy := range proxies {
		err := f(proxy.url)
		if err == nil {
			return nil
		}
		isNotExistErr := errors.Is(err, fs.ErrNotExist)

		if proxy.url == "direct" || (proxy.url == "noproxy" && err != errUseProxy) {
			bestErr = err
			bestErrRank = directRank
		} else if bestErrRank <= proxyRank && !isNotExistErr {
			bestErr = err
			bestErrRank = proxyRank
		} else if bestErrRank == notExistRank {
			bestErr = err
		}

		if !proxy.fallBackOnError && !isNotExistErr {
			break
		}
	}
	return bestErr
}

type proxyRepo struct {
	url          *url.URL // The combined module proxy URL joined with the module path.
	path         string   // The module path (unescaped).
	redactedBase string   // The base module proxy URL in [url.URL.Redacted] form.

	listLatestOnce sync.Once
	listLatest     *RevInfo
	listLatestErr  error
}

func newProxyRepo(baseURL, path string) (Repo, error) {
	// Parse the base proxy URL.
	base, err := url.Parse(baseURL)
	if err != nil {
		return nil, err
	}
	redactedBase := base.Redacted()
	switch base.Scheme {
	case "http", "https":
		// ok
	case "file":
		if *base != (url.URL{Scheme: base.Scheme, Path: base.Path, RawPath: base.RawPath}) {
			return nil, fmt.Errorf("invalid file:// proxy URL with non-path elements: %s", redactedBase)
		}
	case "":
		return nil, fmt.Errorf("invalid proxy URL missing scheme: %s", redactedBase)
	default:
		return nil, fmt.Errorf("invalid proxy URL scheme (must be https, http, file): %s", redactedBase)
	}

	// Append the module path to the URL.
	url := base
	enc, err := module.EscapePath(path)
	if err != nil {
		return nil, err
	}
	url.Path = strings.TrimSuffix(base.Path, "/") + "/" + enc
	url.RawPath = strings.TrimSuffix(base.RawPath, "/") + "/" + pathEscape(enc)

	return &proxyRepo{url, path, redactedBase, sync.Once{}, nil, nil}, nil
}

func (p *proxyRepo) ModulePath() string {
	return p.path
}

var errProxyReuse = fmt.Errorf("proxy does not support CheckReuse")

func (p *proxyRepo) CheckReuse(ctx context.Context, old *codehost.Origin) error {
	return errProxyReuse
}

// versionError returns err wrapped in a ModuleError for p.path.
func (p *proxyRepo) versionError(version string, err error) error {
	if version != "" && version != module.CanonicalVersion(version) {
		return &module.ModuleError{
			Path: p.path,
			Err: &module.InvalidVersionError{
				Version: version,
				Pseudo:  module.IsPseudoVersion(version),
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

func (p *proxyRepo) getBytes(ctx context.Context, path string) ([]byte, error) {
	body, redactedURL, err := p.getBody(ctx, path)
	if err != nil {
		return nil, err
	}
	defer body.Close()

	b, err := io.ReadAll(body)
	if err != nil {
		// net/http doesn't add context to Body read errors, so add it here.
		// (See https://go.dev/issue/52727.)
		return b, &url.Error{Op: "read", URL: redactedURL, Err: err}
	}
	return b, nil
}

func (p *proxyRepo) getBody(ctx context.Context, path string) (r io.ReadCloser, redactedURL string, err error) {
	fullPath := pathpkg.Join(p.url.Path, path)

	target := *p.url
	target.Path = fullPath
	target.RawPath = pathpkg.Join(target.RawPath, pathEscape(path))

	resp, err := web.Get(web.DefaultSecurity, &target)
	if err != nil {
		return nil, "", err
	}
	if err := resp.Err(); err != nil {
		resp.Body.Close()
		return nil, "", err
	}
	return resp.Body, resp.URL, nil
}

func (p *proxyRepo) Versions(ctx context.Context, prefix string) (*Versions, error) {
	data, err := p.getBytes(ctx, "@v/list")
	if err != nil {
		p.listLatestOnce.Do(func() {
			p.listLatest, p.listLatestErr = nil, p.versionError("", err)
		})
		return nil, p.versionError("", err)
	}
	var list []string
	allLine := strings.Split(string(data), "\n")
	for _, line := range allLine {
		f := strings.Fields(line)
		if len(f) >= 1 && semver.IsValid(f[0]) && strings.HasPrefix(f[0], prefix) && !module.IsPseudoVersion(f[0]) {
			list = append(list, f[0])
		}
	}
	p.listLatestOnce.Do(func() {
		p.listLatest, p.listLatestErr = p.latestFromList(ctx, allLine)
	})
	semver.Sort(list)
	return &Versions{List: list}, nil
}

func (p *proxyRepo) latest(ctx context.Context) (*RevInfo, error) {
	p.listLatestOnce.Do(func() {
		data, err := p.getBytes(ctx, "@v/list")
		if err != nil {
			p.listLatestErr = p.versionError("", err)
			return
		}
		list := strings.Split(string(data), "\n")
		p.listLatest, p.listLatestErr = p.latestFromList(ctx, list)
	})
	return p.listLatest, p.listLatestErr
}

func (p *proxyRepo) latestFromList(ctx context.Context, allLine []string) (*RevInfo, error) {
	var (
		bestTime    time.Time
		bestVersion string
	)
	for _, line := range allLine {
		f := strings.Fields(line)
		if len(f) >= 1 && semver.IsValid(f[0]) {
			// If the proxy includes timestamps, prefer the timestamp it reports.
			// Otherwise, derive the timestamp from the pseudo-version.
			var (
				ft time.Time
			)
			if len(f) >= 2 {
				ft, _ = time.Parse(time.RFC3339, f[1])
			} else if module.IsPseudoVersion(f[0]) {
				ft, _ = module.PseudoVersionTime(f[0])
			} else {
				// Repo.Latest promises that this method is only called where there are
				// no tagged versions. Ignore any tagged versions that were added in the
				// meantime.
				continue
			}
			if bestTime.Before(ft) {
				bestTime = ft
				bestVersion = f[0]
			}
		}
	}
	if bestVersion == "" {
		return nil, p.versionError("", codehost.ErrNoCommits)
	}

	// Call Stat to get all the other fields, including Origin information.
	return p.Stat(ctx, bestVersion)
}

func (p *proxyRepo) Stat(ctx context.Context, rev string) (*RevInfo, error) {
	encRev, err := module.EscapeVersion(rev)
	if err != nil {
		return nil, p.versionError(rev, err)
	}
	data, err := p.getBytes(ctx, "@v/"+encRev+".info")
	if err != nil {
		return nil, p.versionError(rev, err)
	}
	info := new(RevInfo)
	if err := json.Unmarshal(data, info); err != nil {
		return nil, p.versionError(rev, fmt.Errorf("invalid response from proxy %q: %w", p.redactedBase, err))
	}
	if info.Version != rev && rev == module.CanonicalVersion(rev) && module.Check(p.path, rev) == nil {
		// If we request a correct, appropriate version for the module path, the
		// proxy must return either exactly that version or an error — not some
		// arbitrary other version.
		return nil, p.versionError(rev, fmt.Errorf("proxy returned info for version %s instead of requested version", info.Version))
	}
	return info, nil
}

func (p *proxyRepo) Latest(ctx context.Context) (*RevInfo, error) {
	data, err := p.getBytes(ctx, "@latest")
	if err != nil {
		if !errors.Is(err, fs.ErrNotExist) {
			return nil, p.versionError("", err)
		}
		return p.latest(ctx)
	}
	info := new(RevInfo)
	if err := json.Unmarshal(data, info); err != nil {
		return nil, p.versionError("", fmt.Errorf("invalid response from proxy %q: %w", p.redactedBase, err))
	}
	return info, nil
}

func (p *proxyRepo) GoMod(ctx context.Context, version string) ([]byte, error) {
	if version != module.CanonicalVersion(version) {
		return nil, p.versionError(version, fmt.Errorf("internal error: version passed to GoMod is not canonical"))
	}

	encVer, err := module.EscapeVersion(version)
	if err != nil {
		return nil, p.versionError(version, err)
	}
	data, err := p.getBytes(ctx, "@v/"+encVer+".mod")
	if err != nil {
		return nil, p.versionError(version, err)
	}
	return data, nil
}

func (p *proxyRepo) Zip(ctx context.Context, dst io.Writer, version string) error {
	if version != module.CanonicalVersion(version) {
		return p.versionError(version, fmt.Errorf("internal error: version passed to Zip is not canonical"))
	}

	encVer, err := module.EscapeVersion(version)
	if err != nil {
		return p.versionError(version, err)
	}
	path := "@v/" + encVer + ".zip"
	body, redactedURL, err := p.getBody(ctx, path)
	if err != nil {
		return p.versionError(version, err)
	}
	defer body.Close()

	lr := &io.LimitedReader{R: body, N: codehost.MaxZipFile + 1}
	if _, err := io.Copy(dst, lr); err != nil {
		// net/http doesn't add context to Body read errors, so add it here.
		// (See https://go.dev/issue/52727.)
		err = &url.Error{Op: "read", URL: redactedURL, Err: err}
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
