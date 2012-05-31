// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

// TODO(adg): test authentication

import (
	"appengine"
	"appengine/datastore"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"time"
)

func init() {
	http.HandleFunc("/buildtest", testHandler)
}

var testEntityKinds = []string{
	"Package",
	"Commit",
	"Result",
	"Log",
}

const testPkg = "code.google.com/p/go.test"

var testPackage = &Package{Name: "Test", Kind: "subrepo", Path: testPkg}

var testPackages = []*Package{
	{Name: "Go", Path: ""},
	testPackage,
}

var tCommitTime = time.Now().Add(-time.Hour * 24 * 7)

func tCommit(hash, parentHash, path string) *Commit {
	tCommitTime.Add(time.Hour) // each commit should have a different time
	return &Commit{
		PackagePath: path,
		Hash:        hash,
		ParentHash:  parentHash,
		Time:        tCommitTime,
		User:        "adg",
		Desc:        "change description " + hash,
	}
}

var testRequests = []struct {
	path string
	vals url.Values
	req  interface{}
	res  interface{}
}{
	// Packages
	{"/packages?kind=subrepo", nil, nil, []*Package{testPackage}},

	// Go repo
	{"/commit", nil, tCommit("0001", "0000", ""), nil},
	{"/commit", nil, tCommit("0002", "0001", ""), nil},
	{"/commit", nil, tCommit("0003", "0002", ""), nil},
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-386"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0003"}}},
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0003"}}},
	{"/result", nil, &Result{Builder: "linux-386", Hash: "0001", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-386"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0003"}}},
	{"/result", nil, &Result{Builder: "linux-386", Hash: "0002", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-386"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0003"}}},

	// multiple builders
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0003"}}},
	{"/result", nil, &Result{Builder: "linux-amd64", Hash: "0003", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-386"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0003"}}},
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0002"}}},

	// branches
	{"/commit", nil, tCommit("0004", "0003", ""), nil},
	{"/commit", nil, tCommit("0005", "0002", ""), nil},
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-386"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0005"}}},
	{"/result", nil, &Result{Builder: "linux-386", Hash: "0005", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-386"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0004"}}},
	{"/result", nil, &Result{Builder: "linux-386", Hash: "0004", OK: false}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-386"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0003"}}},

	// logs
	{"/result", nil, &Result{Builder: "linux-386", Hash: "0003", OK: false, Log: "test"}, nil},
	{"/log/a94a8fe5ccb19ba61c4c0873d391e987982fbbd3", nil, nil, "test"},
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-386"}}, nil, nil},

	// repeat failure (shouldn't re-send mail)
	{"/result", nil, &Result{Builder: "linux-386", Hash: "0003", OK: false, Log: "test"}, nil},

	// non-Go repos
	{"/commit", nil, tCommit("1001", "1000", testPkg), nil},
	{"/commit", nil, tCommit("1002", "1001", testPkg), nil},
	{"/commit", nil, tCommit("1003", "1002", testPkg), nil},
	{"/todo", url.Values{"kind": {"build-package"}, "builder": {"linux-386"}, "packagePath": {testPkg}, "goHash": {"0001"}}, nil, &Todo{Kind: "build-package", Data: &Commit{Hash: "1003"}}},
	{"/result", nil, &Result{PackagePath: testPkg, Builder: "linux-386", Hash: "1003", GoHash: "0001", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-package"}, "builder": {"linux-386"}, "packagePath": {testPkg}, "goHash": {"0001"}}, nil, &Todo{Kind: "build-package", Data: &Commit{Hash: "1002"}}},
	{"/result", nil, &Result{PackagePath: testPkg, Builder: "linux-386", Hash: "1002", GoHash: "0001", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-package"}, "builder": {"linux-386"}, "packagePath": {testPkg}, "goHash": {"0001"}}, nil, &Todo{Kind: "build-package", Data: &Commit{Hash: "1001"}}},
	{"/result", nil, &Result{PackagePath: testPkg, Builder: "linux-386", Hash: "1001", GoHash: "0001", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-package"}, "builder": {"linux-386"}, "packagePath": {testPkg}, "goHash": {"0001"}}, nil, nil},
	{"/todo", url.Values{"kind": {"build-package"}, "builder": {"linux-386"}, "packagePath": {testPkg}, "goHash": {"0002"}}, nil, &Todo{Kind: "build-package", Data: &Commit{Hash: "1003"}}},

	// re-build Go revision for stale subrepos
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-386"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0005"}}},
	{"/result", nil, &Result{PackagePath: testPkg, Builder: "linux-386", Hash: "1001", GoHash: "0005", OK: false, Log: "boo"}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-386"}}, nil, nil},
}

func testHandler(w http.ResponseWriter, r *http.Request) {
	if !appengine.IsDevAppServer() {
		fmt.Fprint(w, "These tests must be run under the dev_appserver.")
		return
	}
	c := appengine.NewContext(r)
	if err := nukeEntities(c, testEntityKinds); err != nil {
		logErr(w, r, err)
		return
	}
	if r.FormValue("nukeonly") != "" {
		fmt.Fprint(w, "OK")
		return
	}

	for _, p := range testPackages {
		if _, err := datastore.Put(c, p.Key(c), p); err != nil {
			logErr(w, r, err)
			return
		}
	}

	for i, t := range testRequests {
		c.Infof("running test %d %s", i, t.path)
		errorf := func(format string, args ...interface{}) {
			fmt.Fprintf(w, "%d %s: ", i, t.path)
			fmt.Fprintf(w, format, args...)
			fmt.Fprintln(w)
		}
		var body io.ReadWriter
		if t.req != nil {
			body = new(bytes.Buffer)
			json.NewEncoder(body).Encode(t.req)
		}
		url := "http://" + domain + t.path
		if t.vals != nil {
			url += "?" + t.vals.Encode()
		}
		req, err := http.NewRequest("POST", url, body)
		if err != nil {
			logErr(w, r, err)
			return
		}
		if t.req != nil {
			req.Method = "POST"
		}
		req.Header = r.Header
		rec := httptest.NewRecorder()

		// Make the request
		http.DefaultServeMux.ServeHTTP(rec, req)

		if rec.Code != 0 && rec.Code != 200 {
			errorf(rec.Body.String())
			return
		}
		resp := new(dashResponse)

		// If we're expecting a *Todo value,
		// prime the Response field with a Todo and a Commit inside it.
		if _, ok := t.res.(*Todo); ok {
			resp.Response = &Todo{Data: &Commit{}}
		}

		if strings.HasPrefix(t.path, "/log/") {
			resp.Response = rec.Body.String()
		} else {
			err := json.NewDecoder(rec.Body).Decode(resp)
			if err != nil {
				errorf("decoding response: %v", err)
				return
			}
		}
		if e, ok := t.res.(string); ok {
			g, ok := resp.Response.(string)
			if !ok {
				errorf("Response not string: %T", resp.Response)
				return
			}
			if g != e {
				errorf("response mismatch: got %q want %q", g, e)
				return
			}
		}
		if e, ok := t.res.(*Todo); ok {
			g, ok := resp.Response.(*Todo)
			if !ok {
				errorf("Response not *Todo: %T", resp.Response)
				return
			}
			if e.Data == nil && g.Data != nil {
				errorf("Response.Data should be nil, got: %v", g.Data)
				return
			}
			if g.Data == nil {
				errorf("Response.Data is nil, want: %v", e.Data)
				return
			}
			gd, ok := g.Data.(*Commit)
			if !ok {
				errorf("Response.Data not *Commit: %T", g.Data)
				return
			}
			if eh := e.Data.(*Commit).Hash; eh != gd.Hash {
				errorf("hashes don't match: got %q, want %q", gd.Hash, eh)
				return
			}
		}
		if t.res == nil && resp.Response != nil {
			errorf("response mismatch: got %q expected <nil>",
				resp.Response)
			return
		}
	}
	fmt.Fprint(w, "PASS\nYou should see only one mail notification (for 0003/linux-386) in the dev_appserver logs.")
}

func nukeEntities(c appengine.Context, kinds []string) error {
	if !appengine.IsDevAppServer() {
		return errors.New("can't nuke production data")
	}
	var keys []*datastore.Key
	for _, kind := range kinds {
		q := datastore.NewQuery(kind).KeysOnly()
		for t := q.Run(c); ; {
			k, err := t.Next(nil)
			if err == datastore.Done {
				break
			}
			if err != nil {
				return err
			}
			keys = append(keys, k)
		}
	}
	return datastore.DeleteMulti(c, keys)
}
