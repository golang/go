// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"errors"
	"io/ioutil"
	"net/http"
	"testing"
)

var FindPublicRepoTests = []struct {
	pkg            string
	vcs, root, url string
	transport      *testTransport
}{
	{
		"code.google.com/p/repo/path/foo",
		"hg",
		"code.google.com/p/repo",
		"https://code.google.com/p/repo",
		&testTransport{
			"https://code.google.com/p/repo/source/checkout",
			`<tt id="checkoutcmd">hg clone https://...`,
		},
	},
	{
		"code.google.com/p/repo/path/foo",
		"svn",
		"code.google.com/p/repo",
		"https://code.google.com/p/repo",
		&testTransport{
			"https://code.google.com/p/repo/source/checkout",
			`<tt id="checkoutcmd">svn checkout https://...`,
		},
	},
	{
		"code.google.com/p/repo/path/foo",
		"git",
		"code.google.com/p/repo",
		"https://code.google.com/p/repo",
		&testTransport{
			"https://code.google.com/p/repo/source/checkout",
			`<tt id="checkoutcmd">git clone https://...`,
		},
	},
	{
		"code.google.com/p/repo.sub/path",
		"hg",
		"code.google.com/p/repo.sub",
		"https://code.google.com/p/repo.sub",
		&testTransport{
			"https://code.google.com/p/repo/source/checkout?repo=sub",
			`<tt id="checkoutcmd">hg clone https://...`,
		},
	},
	{
		"bitbucket.org/user/repo/path/foo",
		"hg",
		"bitbucket.org/user/repo",
		"http://bitbucket.org/user/repo",
		&testTransport{
			"https://api.bitbucket.org/1.0/repositories/user/repo",
			`{"scm": "hg"}`,
		},
	},
	{
		"bitbucket.org/user/repo/path/foo",
		"git",
		"bitbucket.org/user/repo",
		"http://bitbucket.org/user/repo.git",
		&testTransport{
			"https://api.bitbucket.org/1.0/repositories/user/repo",
			`{"scm": "git"}`,
		},
	},
	{
		"github.com/user/repo/path/foo",
		"git",
		"github.com/user/repo",
		"http://github.com/user/repo.git",
		nil,
	},
	{
		"launchpad.net/project/series/path",
		"bzr",
		"launchpad.net/project/series",
		"https://launchpad.net/project/series",
		nil,
	},
	{
		"launchpad.net/~user/project/branch/path",
		"bzr",
		"launchpad.net/~user/project/branch",
		"https://launchpad.net/~user/project/branch",
		nil,
	},
}

func TestFindPublicRepo(t *testing.T) {
	for _, test := range FindPublicRepoTests {
		client := http.DefaultClient
		if test.transport != nil {
			client = &http.Client{Transport: test.transport}
		}
		repo, err := findPublicRepo(test.pkg)
		if err != nil {
			t.Errorf("findPublicRepo(%s): error: %v", test.pkg, err)
			continue
		}
		if repo == nil {
			t.Errorf("%s: got nil match", test.pkg)
			continue
		}
		url, root, vcs, err := repo.Repo(client)
		if err != nil {
			t.Errorf("%s: repo.Repo error: %v", test.pkg, err)
			continue
		}
		if v := vcsMap[test.vcs]; vcs != v {
			t.Errorf("%s: got vcs=%v, want %v", test.pkg, vcs, v)
		}
		if root != test.root {
			t.Errorf("%s: got root=%v, want %v", test.pkg, root, test.root)
		}
		if url != test.url {
			t.Errorf("%s: got url=%v, want %v", test.pkg, url, test.url)
		}
	}
}

type testTransport struct {
	expectURL    string
	responseBody string
}

func (t *testTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if g, e := req.URL.String(), t.expectURL; g != e {
		return nil, errors.New("want " + e)
	}
	body := ioutil.NopCloser(bytes.NewBufferString(t.responseBody))
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       body,
	}, nil
}
