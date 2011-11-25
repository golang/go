// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

// TODO(adg): test branches
// TODO(adg): test non-Go packages
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

var testRequests = []struct {
	path string
	vals url.Values
	req  interface{}
	res  interface{}
}{
	{"/commit", nil, &Commit{Hash: "0001", ParentHash: "0000"}, nil},
	{"/commit", nil, &Commit{Hash: "0002", ParentHash: "0001"}, nil},
	{"/commit", nil, &Commit{Hash: "0003", ParentHash: "0002"}, nil},
	{"/todo", url.Values{"builder": {"linux-386"}}, nil, "0003"},
	{"/todo", url.Values{"builder": {"linux-amd64"}}, nil, "0003"},

	{"/result", nil, &Result{Builder: "linux-386", Hash: "0001", OK: true}, nil},
	{"/todo", url.Values{"builder": {"linux-386"}}, nil, "0003"},

	{"/result", nil, &Result{Builder: "linux-386", Hash: "0002", OK: false, Log: []byte("test")}, nil},
	{"/todo", url.Values{"builder": {"linux-386"}}, nil, "0003"},
	{"/log/a94a8fe5ccb19ba61c4c0873d391e987982fbbd3", nil, nil, "test"},

	{"/result", nil, &Result{Builder: "linux-amd64", Hash: "0003", OK: true}, nil},
	{"/todo", url.Values{"builder": {"linux-386"}}, nil, "0003"},
	{"/todo", url.Values{"builder": {"linux-amd64"}}, nil, "0002"},
}

var testPackages = []*Package{
	&Package{Name: "Go", Path: ""},
	&Package{Name: "Other", Path: "code.google.com/p/go.other"},
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

	for _, p := range testPackages {
		if _, err := datastore.Put(c, p.Key(c), p); err != nil {
			logErr(w, r, err)
			return
		}
	}

	failed := false
	for i, t := range testRequests {
		errorf := func(format string, args ...interface{}) {
			fmt.Fprintf(w, "%d %s: ", i, t.path)
			fmt.Fprintf(w, format, args...)
			fmt.Fprintln(w)
			failed = true
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
		req.Header = r.Header
		rec := httptest.NewRecorder()
		http.DefaultServeMux.ServeHTTP(rec, req)
		if rec.Code != 0 && rec.Code != 200 {
			errorf(rec.Body.String())
		}
		if e, ok := t.res.(string); ok {
			g := rec.Body.String()
			if g != e {
				errorf("body mismatch: got %q want %q", g, e)
			}
		}
	}
	if !failed {
		fmt.Fprint(w, "PASS")
	}
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
