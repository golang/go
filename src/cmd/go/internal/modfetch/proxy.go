// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/url"
	"os"
	"strings"
	"time"

	"cmd/go/internal/base"
	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/module"
	"cmd/go/internal/semver"
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
any source. Otherwise, GOPROXY is expected to be the URL of a module proxy,
in which case the go command will fetch all modules from that proxy.
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

var proxyURL = os.Getenv("GOPROXY")

func lookupProxy(path string) (Repo, error) {
	if strings.Contains(proxyURL, ",") {
		return nil, fmt.Errorf("invalid $GOPROXY setting: cannot have comma")
	}
	u, err := url.Parse(proxyURL)
	if err != nil || u.Scheme != "http" && u.Scheme != "https" && u.Scheme != "file" {
		// Don't echo $GOPROXY back in case it has user:password in it (sigh).
		return nil, fmt.Errorf("invalid $GOPROXY setting: malformed URL or invalid scheme (must be http, https, file)")
	}
	return newProxyRepo(u.String(), path)
}

type proxyRepo struct {
	url  string
	path string
}

func newProxyRepo(baseURL, path string) (Repo, error) {
	enc, err := module.EncodePath(path)
	if err != nil {
		return nil, err
	}
	return &proxyRepo{strings.TrimSuffix(baseURL, "/") + "/" + pathEscape(enc), path}, nil
}

func (p *proxyRepo) ModulePath() string {
	return p.path
}

func (p *proxyRepo) Versions(prefix string) ([]string, error) {
	var data []byte
	err := webGetBytes(p.url+"/@v/list", &data)
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
	var data []byte
	err := webGetBytes(p.url+"/@v/list", &data)
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
	var data []byte
	encRev, err := module.EncodeVersion(rev)
	if err != nil {
		return nil, err
	}
	err = webGetBytes(p.url+"/@v/"+pathEscape(encRev)+".info", &data)
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
	var data []byte
	u := p.url + "/@latest"
	err := webGetBytes(u, &data)
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
	var data []byte
	encVer, err := module.EncodeVersion(version)
	if err != nil {
		return nil, err
	}
	err = webGetBytes(p.url+"/@v/"+pathEscape(encVer)+".mod", &data)
	if err != nil {
		return nil, err
	}
	return data, nil
}

func (p *proxyRepo) Zip(version string, tmpdir string) (tmpfile string, err error) {
	var body io.ReadCloser
	encVer, err := module.EncodeVersion(version)
	if err != nil {
		return "", err
	}
	err = webGetBody(p.url+"/@v/"+pathEscape(encVer)+".zip", &body)
	if err != nil {
		return "", err
	}
	defer body.Close()

	// Spool to local file.
	f, err := ioutil.TempFile(tmpdir, "go-proxy-download-")
	if err != nil {
		return "", err
	}
	defer f.Close()
	maxSize := int64(codehost.MaxZipFile)
	lr := &io.LimitedReader{R: body, N: maxSize + 1}
	if _, err := io.Copy(f, lr); err != nil {
		os.Remove(f.Name())
		return "", err
	}
	if lr.N <= 0 {
		os.Remove(f.Name())
		return "", fmt.Errorf("downloaded zip file too large")
	}
	if err := f.Close(); err != nil {
		os.Remove(f.Name())
		return "", err
	}
	return f.Name(), nil
}

// pathEscape escapes s so it can be used in a path.
// That is, it escapes things like ? and # (which really shouldn't appear anyway).
// It does not escape / to %2F: our REST API is designed so that / can be left as is.
func pathEscape(s string) string {
	return strings.Replace(url.PathEscape(s), "%2F", "/", -1)
}
