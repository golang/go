// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build appengine

package build

// TODO(adg): test authentication
// TODO(adg): refactor to use appengine/aetest instead

import (
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

	"appengine"
	"appengine/datastore"
)

func init() {
	handleFunc("/buildtest", testHandler)
}

var testEntityKinds = []string{
	"Package",
	"Commit",
	"CommitRun",
	"Result",
	"PerfResult",
	"PerfMetricRun",
	"PerfConfig",
	"PerfTodo",
	"Log",
}

const testPkg = "golang.org/x/test"

var testPackage = &Package{Name: "Test", Kind: "subrepo", Path: testPkg}

var testPackages = []*Package{
	{Name: "Go", Path: ""},
	testPackage,
}

var tCommitTime = time.Now().Add(-time.Hour * 24 * 7)

func tCommit(hash, parentHash, path string, bench bool) *Commit {
	tCommitTime.Add(time.Hour) // each commit should have a different time
	return &Commit{
		PackagePath:       path,
		Hash:              hash,
		ParentHash:        parentHash,
		Time:              tCommitTime,
		User:              "adg",
		Desc:              "change description " + hash,
		NeedsBenchmarking: bench,
	}
}

var testRequests = []struct {
	path string
	vals url.Values
	req  interface{}
	res  interface{}
}{
	// Packages
	{"/packages", url.Values{"kind": {"subrepo"}}, nil, []*Package{testPackage}},

	// Go repo
	{"/commit", nil, tCommit("0001", "0000", "", true), nil},
	{"/commit", nil, tCommit("0002", "0001", "", false), nil},
	{"/commit", nil, tCommit("0003", "0002", "", true), nil},
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-386"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0003"}}},
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0003"}}},
	{"/result", nil, &Result{Builder: "linux-386", Hash: "0001", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-386"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0003"}}},
	{"/result", nil, &Result{Builder: "linux-386", Hash: "0002", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-386"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0003"}}},

	// Other builders, to test the UI.
	{"/result", nil, &Result{Builder: "linux-amd64", Hash: "0001", OK: true}, nil},
	{"/result", nil, &Result{Builder: "linux-amd64-race", Hash: "0001", OK: true}, nil},
	{"/result", nil, &Result{Builder: "netbsd-386", Hash: "0001", OK: true}, nil},
	{"/result", nil, &Result{Builder: "plan9-386", Hash: "0001", OK: true}, nil},
	{"/result", nil, &Result{Builder: "windows-386", Hash: "0001", OK: true}, nil},
	{"/result", nil, &Result{Builder: "windows-amd64", Hash: "0001", OK: true}, nil},
	{"/result", nil, &Result{Builder: "windows-amd64-race", Hash: "0001", OK: true}, nil},
	{"/result", nil, &Result{Builder: "linux-amd64-temp", Hash: "0001", OK: true}, nil},

	// multiple builders
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0003"}}},
	{"/result", nil, &Result{Builder: "linux-amd64", Hash: "0003", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-386"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0003"}}},
	{"/todo", url.Values{"kind": {"build-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0002"}}},

	// branches
	{"/commit", nil, tCommit("0004", "0003", "", false), nil},
	{"/commit", nil, tCommit("0005", "0002", "", false), nil},
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
	{"/commit", nil, tCommit("1001", "0000", testPkg, false), nil},
	{"/commit", nil, tCommit("1002", "1001", testPkg, false), nil},
	{"/commit", nil, tCommit("1003", "1002", testPkg, false), nil},
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

	// benchmarks
	// build-go-commit must have precedence over benchmark-go-commit
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0005"}}},
	// drain build-go-commit todo
	{"/result", nil, &Result{Builder: "linux-amd64", Hash: "0005", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0004"}}},
	{"/result", nil, &Result{Builder: "linux-amd64", Hash: "0004", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0002"}}},
	{"/result", nil, &Result{Builder: "linux-amd64", Hash: "0002", OK: true}, nil},
	// drain sub-repo todos
	{"/result", nil, &Result{PackagePath: testPkg, Builder: "linux-amd64", Hash: "1001", GoHash: "0005", OK: false}, nil},
	{"/result", nil, &Result{PackagePath: testPkg, Builder: "linux-amd64", Hash: "1002", GoHash: "0005", OK: false}, nil},
	{"/result", nil, &Result{PackagePath: testPkg, Builder: "linux-amd64", Hash: "1003", GoHash: "0005", OK: false}, nil},
	// now we must get benchmark todo
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "benchmark-go-commit", Data: &Commit{Hash: "0003", PerfResults: []string{}}}},
	{"/perf-result", nil, &PerfRequest{Builder: "linux-amd64", Benchmark: "http", Hash: "0003", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "benchmark-go-commit", Data: &Commit{Hash: "0003", PerfResults: []string{"http"}}}},
	{"/perf-result", nil, &PerfRequest{Builder: "linux-amd64", Benchmark: "json", Hash: "0003", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "benchmark-go-commit", Data: &Commit{Hash: "0003", PerfResults: []string{"http", "json"}}}},
	{"/perf-result", nil, &PerfRequest{Builder: "linux-amd64", Benchmark: "meta-done", Hash: "0003", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "benchmark-go-commit", Data: &Commit{Hash: "0001", PerfResults: []string{}}}},
	{"/perf-result", nil, &PerfRequest{Builder: "linux-amd64", Benchmark: "http", Hash: "0001", OK: true}, nil},
	{"/perf-result", nil, &PerfRequest{Builder: "linux-amd64", Benchmark: "meta-done", Hash: "0001", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-amd64"}}, nil, nil},
	// create new commit, it must appear in todo
	{"/commit", nil, tCommit("0006", "0005", "", true), nil},
	// drain build-go-commit todo
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0006"}}},
	{"/result", nil, &Result{Builder: "linux-amd64", Hash: "0006", OK: true}, nil},
	{"/result", nil, &Result{PackagePath: testPkg, Builder: "linux-amd64", Hash: "1003", GoHash: "0006", OK: false}, nil},
	{"/result", nil, &Result{PackagePath: testPkg, Builder: "linux-amd64", Hash: "1002", GoHash: "0006", OK: false}, nil},
	{"/result", nil, &Result{PackagePath: testPkg, Builder: "linux-amd64", Hash: "1001", GoHash: "0006", OK: false}, nil},
	// now we must get benchmark todo
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "benchmark-go-commit", Data: &Commit{Hash: "0006", PerfResults: []string{}}}},
	{"/perf-result", nil, &PerfRequest{Builder: "linux-amd64", Benchmark: "http", Hash: "0006", OK: true}, nil},
	{"/perf-result", nil, &PerfRequest{Builder: "linux-amd64", Benchmark: "meta-done", Hash: "0006", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-amd64"}}, nil, nil},
	// create new benchmark, all commits must re-appear in todo
	{"/commit", nil, tCommit("0007", "0006", "", true), nil},
	// drain build-go-commit todo
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0007"}}},
	{"/result", nil, &Result{Builder: "linux-amd64", Hash: "0007", OK: true}, nil},
	{"/result", nil, &Result{PackagePath: testPkg, Builder: "linux-amd64", Hash: "1003", GoHash: "0007", OK: false}, nil},
	{"/result", nil, &Result{PackagePath: testPkg, Builder: "linux-amd64", Hash: "1002", GoHash: "0007", OK: false}, nil},
	{"/result", nil, &Result{PackagePath: testPkg, Builder: "linux-amd64", Hash: "1001", GoHash: "0007", OK: false}, nil},
	// now we must get benchmark todo
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "benchmark-go-commit", Data: &Commit{Hash: "0007", PerfResults: []string{}}}},
	{"/perf-result", nil, &PerfRequest{Builder: "linux-amd64", Benchmark: "bson", Hash: "0007", OK: true}, nil},
	{"/perf-result", nil, &PerfRequest{Builder: "linux-amd64", Benchmark: "meta-done", Hash: "0007", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "benchmark-go-commit", Data: &Commit{Hash: "0007", PerfResults: []string{"bson"}}}},
	{"/perf-result", nil, &PerfRequest{Builder: "linux-amd64", Benchmark: "meta-done", Hash: "0007", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "benchmark-go-commit", Data: &Commit{Hash: "0006", PerfResults: []string{"http"}}}},
	{"/perf-result", nil, &PerfRequest{Builder: "linux-amd64", Benchmark: "meta-done", Hash: "0006", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "benchmark-go-commit", Data: &Commit{Hash: "0001", PerfResults: []string{"http"}}}},
	{"/perf-result", nil, &PerfRequest{Builder: "linux-amd64", Benchmark: "meta-done", Hash: "0001", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-amd64"}}, nil, &Todo{Kind: "benchmark-go-commit", Data: &Commit{Hash: "0003", PerfResults: []string{"http", "json"}}}},
	{"/perf-result", nil, &PerfRequest{Builder: "linux-amd64", Benchmark: "meta-done", Hash: "0003", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-amd64"}}, nil, nil},
	// attach second builder
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-386"}}, nil, &Todo{Kind: "build-go-commit", Data: &Commit{Hash: "0007"}}},
	// drain build-go-commit todo
	{"/result", nil, &Result{Builder: "linux-386", Hash: "0007", OK: true}, nil},
	{"/result", nil, &Result{Builder: "linux-386", Hash: "0006", OK: true}, nil},
	{"/result", nil, &Result{PackagePath: testPkg, Builder: "linux-386", Hash: "1003", GoHash: "0007", OK: false}, nil},
	{"/result", nil, &Result{PackagePath: testPkg, Builder: "linux-386", Hash: "1002", GoHash: "0007", OK: false}, nil},
	{"/result", nil, &Result{PackagePath: testPkg, Builder: "linux-386", Hash: "1001", GoHash: "0007", OK: false}, nil},
	// now we must get benchmark todo
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-386"}}, nil, &Todo{Kind: "benchmark-go-commit", Data: &Commit{Hash: "0007"}}},
	{"/perf-result", nil, &PerfRequest{Builder: "linux-386", Benchmark: "meta-done", Hash: "0007", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-386"}}, nil, &Todo{Kind: "benchmark-go-commit", Data: &Commit{Hash: "0006"}}},
	{"/perf-result", nil, &PerfRequest{Builder: "linux-386", Benchmark: "meta-done", Hash: "0006", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-386"}}, nil, &Todo{Kind: "benchmark-go-commit", Data: &Commit{Hash: "0001"}}},
	{"/perf-result", nil, &PerfRequest{Builder: "linux-386", Benchmark: "meta-done", Hash: "0001", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-386"}}, nil, &Todo{Kind: "benchmark-go-commit", Data: &Commit{Hash: "0003"}}},
	{"/perf-result", nil, &PerfRequest{Builder: "linux-386", Benchmark: "meta-done", Hash: "0003", OK: true}, nil},
	{"/todo", url.Values{"kind": {"build-go-commit", "benchmark-go-commit"}, "builder": {"linux-386"}}, nil, nil},
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

	origReq := *r
	defer func() {
		// HACK: We need to clobber the original request (see below)
		// so make sure we fix it before exiting the handler.
		*r = origReq
	}()
	for i, t := range testRequests {
		c.Infof("running test %d %s vals='%q' req='%q' res='%q'", i, t.path, t.vals, t.req, t.res)
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
			url += "?" + t.vals.Encode() + "&version=2"
		} else {
			url += "?version=2"
		}
		req, err := http.NewRequest("POST", url, body)
		if err != nil {
			logErr(w, r, err)
			return
		}
		if t.req != nil {
			req.Method = "POST"
		}
		req.Header = origReq.Header
		rec := httptest.NewRecorder()

		// Make the request
		*r = *req // HACK: App Engine uses the request pointer
		// as a map key to resolve Contexts.
		http.DefaultServeMux.ServeHTTP(rec, r)

		if rec.Code != 0 && rec.Code != 200 {
			errorf(rec.Body.String())
			return
		}
		c.Infof("response='%v'", rec.Body.String())
		resp := new(dashResponse)

		// If we're expecting a *Todo value,
		// prime the Response field with a Todo and a Commit inside it.
		if t.path == "/todo" {
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
			if g.Kind != e.Kind {
				errorf("kind don't match: got %q, want %q", g.Kind, e.Kind)
				return
			}
			ed := e.Data.(*Commit)
			if ed.Hash != gd.Hash {
				errorf("hashes don't match: got %q, want %q", gd.Hash, ed.Hash)
				return
			}
			if len(gd.PerfResults) != len(ed.PerfResults) {
				errorf("result data len don't match: got %v, want %v", len(gd.PerfResults), len(ed.PerfResults))
				return
			}
			for i := range gd.PerfResults {
				if gd.PerfResults[i] != ed.PerfResults[i] {
					errorf("result data %v don't match: got %v, want %v", i, gd.PerfResults[i], ed.PerfResults[i])
					return
				}
			}
		}
		if t.res == nil && resp.Response != nil {
			errorf("response mismatch: got %q expected <nil>", resp.Response)
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
