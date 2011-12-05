// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

// TODO(adg): test authentication

import (
	"appengine"
	"appengine/datastore"
	"bytes"
	"fmt"
	"http"
	"http/httptest"
	"io"
	"json"
	"os"
	"strings"
	"url"
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

var testPackage = &Package{Name: "Test", Path: testPkg}

var testPackages = []*Package{
	&Package{Name: "Go", Path: ""},
	testPackage,
}

var testRequests = []struct {
	path string
	vals url.Values
	req  interface{}
	res  interface{}
}{
	// Packages
	{"/packages", nil, nil, []*Package{testPackage}},

	// Go repo
	{"/commit", nil, &Commit{Hash: "0001", ParentHash: "0000"}, nil},
	{"/commit", nil, &Commit{Hash: "0002", ParentHash: "0001"}, nil},
	{"/commit", nil, &Commit{Hash: "0003", ParentHash: "0002"}, nil},
	{"/todo", url.Values{"builder": {"linux-386"}}, nil, "0003"},
	{"/todo", url.Values{"builder": {"linux-amd64"}}, nil, "0003"},
	{"/result", nil, &Result{Builder: "linux-386", Hash: "0001", OK: true}, nil},
	{"/todo", url.Values{"builder": {"linux-386"}}, nil, "0003"},
	{"/result", nil, &Result{Builder: "linux-386", Hash: "0002", OK: true}, nil},
	{"/todo", url.Values{"builder": {"linux-386"}}, nil, "0003"},

	// multiple builders
	{"/todo", url.Values{"builder": {"linux-amd64"}}, nil, "0003"},
	{"/result", nil, &Result{Builder: "linux-amd64", Hash: "0003", OK: true}, nil},
	{"/todo", url.Values{"builder": {"linux-386"}}, nil, "0003"},
	{"/todo", url.Values{"builder": {"linux-amd64"}}, nil, "0002"},

	// branches
	{"/commit", nil, &Commit{Hash: "0004", ParentHash: "0003"}, nil},
	{"/commit", nil, &Commit{Hash: "0005", ParentHash: "0002"}, nil},
	{"/todo", url.Values{"builder": {"linux-386"}}, nil, "0005"},
	{"/result", nil, &Result{Builder: "linux-386", Hash: "0005", OK: true}, nil},
	{"/todo", url.Values{"builder": {"linux-386"}}, nil, "0004"},
	{"/result", nil, &Result{Builder: "linux-386", Hash: "0004", OK: true}, nil},
	{"/todo", url.Values{"builder": {"linux-386"}}, nil, "0003"},

	// logs
	{"/result", nil, &Result{Builder: "linux-386", Hash: "0003", OK: false, Log: []byte("test")}, nil},
	{"/log/a94a8fe5ccb19ba61c4c0873d391e987982fbbd3", nil, nil, "test"},
	{"/todo", url.Values{"builder": {"linux-386"}}, nil, nil},

	// non-Go repos
	{"/commit", nil, &Commit{PackagePath: testPkg, Hash: "1001", ParentHash: "1000"}, nil},
	{"/commit", nil, &Commit{PackagePath: testPkg, Hash: "1002", ParentHash: "1001"}, nil},
	{"/commit", nil, &Commit{PackagePath: testPkg, Hash: "1003", ParentHash: "1002"}, nil},
	{"/todo", url.Values{"builder": {"linux-386"}, "packagePath": {testPkg}, "goHash": {"0001"}}, nil, "1003"},
	{"/result", nil, &Result{PackagePath: testPkg, Builder: "linux-386", Hash: "1003", GoHash: "0001", OK: true}, nil},
	{"/todo", url.Values{"builder": {"linux-386"}, "packagePath": {testPkg}, "goHash": {"0001"}}, nil, "1002"},
	{"/result", nil, &Result{PackagePath: testPkg, Builder: "linux-386", Hash: "1002", GoHash: "0001", OK: true}, nil},
	{"/todo", url.Values{"builder": {"linux-386"}, "packagePath": {testPkg}, "goHash": {"0001"}}, nil, "1001"},
	{"/result", nil, &Result{PackagePath: testPkg, Builder: "linux-386", Hash: "1001", GoHash: "0001", OK: true}, nil},
	{"/todo", url.Values{"builder": {"linux-386"}, "packagePath": {testPkg}, "goHash": {"0001"}}, nil, nil},
	{"/todo", url.Values{"builder": {"linux-386"}, "packagePath": {testPkg}, "goHash": {"0002"}}, nil, "1003"},
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
		url := "http://" + appengine.DefaultVersionHostname(c) + t.path
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
		http.DefaultServeMux.ServeHTTP(rec, req)
		if rec.Code != 0 && rec.Code != 200 {
			errorf(rec.Body.String())
			return
		}
		resp := new(dashResponse)
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
		if t.res == nil && resp.Response != nil {
			errorf("response mismatch: got %q expected <nil>",
				resp.Response)
			return
		}
	}
	fmt.Fprint(w, "PASS")
}

func nukeEntities(c appengine.Context, kinds []string) os.Error {
	if !appengine.IsDevAppServer() {
		return os.NewError("can't nuke production data")
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
