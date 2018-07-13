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

	"cmd/go/internal/modfetch/codehost"
	"cmd/go/internal/semver"
)

var proxyURL = os.Getenv("GOPROXY")

func lookupProxy(path string) (Repo, error) {
	u, err := url.Parse(proxyURL)
	if err != nil || u.Scheme != "http" && u.Scheme != "https" && u.Scheme != "file" {
		// Don't echo $GOPROXY back in case it has user:password in it (sigh).
		return nil, fmt.Errorf("invalid $GOPROXY setting")
	}
	return newProxyRepo(u.String(), path), nil
}

type proxyRepo struct {
	url  string
	path string
}

func newProxyRepo(baseURL, path string) Repo {
	return &proxyRepo{strings.TrimSuffix(baseURL, "/") + "/" + pathEscape(path), path}
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
	err := webGetBytes(p.url+"/@v/"+pathEscape(rev)+".info", &data)
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
	err := webGetBytes(p.url+"/@v/"+pathEscape(version)+".mod", &data)
	if err != nil {
		return nil, err
	}
	return data, nil
}

func (p *proxyRepo) Zip(version string, tmpdir string) (tmpfile string, err error) {
	var body io.ReadCloser
	err = webGetBody(p.url+"/@v/"+pathEscape(version)+".zip", &body)
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
