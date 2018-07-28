// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfetch

import (
	"internal/testenv"
	"io/ioutil"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cmd/go/internal/module"
)

var downloadZipTestCases = []struct {
	name    string
	path    string
	version string
	err     bool
	pm      func(p, v string) *proxyServer
}{
	{
		name:    "happy path",
		path:    "github.com/rsc/vgotest1",
		version: "v1.0.0",
	},
	{
		name:    "incorrect zip format",
		path:    "github.com/rsc/vgotest1/v2",
		version: "v2.0.0",
		err:     true,
		pm:      badProxy,
	},
}

func TestDownloadZip(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tmpdir, err := ioutil.TempDir("", "gopath")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	for _, tc := range downloadZipTestCases {
		t.Run(tc.name, func(t *testing.T) {
			p := strings.Replace(tc.path, "/", "_", -1)
			zipPath := filepath.Join(tmpdir, p+".zip")
			v := module.Version{Path: tc.path, Version: tc.version}
			if tc.pm != nil {
				s := setProxy(tc.pm(tc.path, tc.version))
				defer func() { s.Close(); proxyURL = "" }()
			}

			err := downloadZip(v, zipPath)
			if tc.err && err == nil {
				t.Fatalf("expected an error to occur but downloadZip returned nil")
			} else if !tc.err && err != nil {
				t.Fatal(err)
			}
		})
	}
}

func badProxy(p, v string) *proxyServer {
	return &proxyServer{zip: []byte("bad zip format")}
}

func setProxy(ps *proxyServer) *httptest.Server {
	s := httptest.NewServer(ps)
	proxyURL = s.URL
	return s
}
