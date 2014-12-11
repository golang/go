// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"time"
)

const builderVersion = 1 // keep in sync with dashboard/app/build/handler.go

type obj map[string]interface{}

// dash runs the given method and command on the dashboard.
// If args is non-nil it is encoded as the URL query string.
// If req is non-nil it is JSON-encoded and passed as the body of the HTTP POST.
// If resp is non-nil the server's response is decoded into the value pointed
// to by resp (resp must be a pointer).
func dash(meth, cmd string, args url.Values, req, resp interface{}) error {
	argsCopy := url.Values{"version": {fmt.Sprint(builderVersion)}}
	for k, v := range args {
		if k == "version" {
			panic(`dash: reserved args key: "version"`)
		}
		argsCopy[k] = v
	}
	var r *http.Response
	var err error
	if *verbose {
		log.Println("dash <-", meth, cmd, argsCopy, req)
	}
	cmd = *dashboard + "/" + cmd + "?" + argsCopy.Encode()
	switch meth {
	case "GET":
		if req != nil {
			log.Panicf("%s to %s with req", meth, cmd)
		}
		r, err = http.Get(cmd)
	case "POST":
		var body io.Reader
		if req != nil {
			b, err := json.Marshal(req)
			if err != nil {
				return err
			}
			body = bytes.NewBuffer(b)
		}
		r, err = http.Post(cmd, "text/json", body)
	default:
		log.Panicf("%s: invalid method %q", cmd, meth)
		panic("invalid method: " + meth)
	}
	if err != nil {
		return err
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		return fmt.Errorf("bad http response: %v", r.Status)
	}
	body := new(bytes.Buffer)
	if _, err := body.ReadFrom(r.Body); err != nil {
		return err
	}

	// Read JSON-encoded Response into provided resp
	// and return an error if present.
	var result = struct {
		Response interface{}
		Error    string
	}{
		// Put the provided resp in here as it can be a pointer to
		// some value we should unmarshal into.
		Response: resp,
	}
	if err = json.Unmarshal(body.Bytes(), &result); err != nil {
		log.Printf("json unmarshal %#q: %s\n", body.Bytes(), err)
		return err
	}
	if *verbose {
		log.Println("dash ->", result)
	}
	if result.Error != "" {
		return errors.New(result.Error)
	}

	return nil
}

// todo returns the next hash to build or benchmark.
func (b *Builder) todo(kinds []string, pkg, goHash string) (kind, rev string, benchs []string, err error) {
	args := url.Values{
		"builder":     {b.name},
		"packagePath": {pkg},
		"goHash":      {goHash},
	}
	for _, k := range kinds {
		args.Add("kind", k)
	}
	var resp *struct {
		Kind string
		Data struct {
			Hash        string
			PerfResults []string
		}
	}
	if err = dash("GET", "todo", args, nil, &resp); err != nil {
		return
	}
	if resp == nil {
		return
	}
	if *verbose {
		fmt.Printf("dash resp: %+v\n", *resp)
	}
	for _, k := range kinds {
		if k == resp.Kind {
			return resp.Kind, resp.Data.Hash, resp.Data.PerfResults, nil
		}
	}
	err = fmt.Errorf("expecting Kinds %q, got %q", kinds, resp.Kind)
	return
}

// recordResult sends build results to the dashboard
func (b *Builder) recordResult(ok bool, pkg, hash, goHash, buildLog string, runTime time.Duration) error {
	if !*report {
		return nil
	}
	req := obj{
		"Builder":     b.name,
		"PackagePath": pkg,
		"Hash":        hash,
		"GoHash":      goHash,
		"OK":          ok,
		"Log":         buildLog,
		"RunTime":     runTime,
	}
	args := url.Values{"key": {b.key}, "builder": {b.name}}
	return dash("POST", "result", args, req, nil)
}

// Result of running a single benchmark on a single commit.
type PerfResult struct {
	Builder   string
	Benchmark string
	Hash      string
	OK        bool
	Metrics   []PerfMetric
	Artifacts []PerfArtifact
}

type PerfMetric struct {
	Type string
	Val  uint64
}

type PerfArtifact struct {
	Type string
	Body string
}

// recordPerfResult sends benchmarking results to the dashboard
func (b *Builder) recordPerfResult(req *PerfResult) error {
	if !*report {
		return nil
	}
	req.Builder = b.name
	args := url.Values{"key": {b.key}, "builder": {b.name}}
	return dash("POST", "perf-result", args, req, nil)
}

func postCommit(key, pkg string, l *HgLog) error {
	if !*report {
		return nil
	}
	t, err := time.Parse(time.RFC3339, l.Date)
	if err != nil {
		return fmt.Errorf("parsing %q: %v", l.Date, t)
	}
	return dash("POST", "commit", url.Values{"key": {key}}, obj{
		"PackagePath":       pkg,
		"Hash":              l.Hash,
		"ParentHash":        l.Parent,
		"Time":              t.Format(time.RFC3339),
		"User":              l.Author,
		"Desc":              l.Desc,
		"NeedsBenchmarking": l.bench,
	}, nil)
}

func dashboardCommit(pkg, hash string) bool {
	err := dash("GET", "commit", url.Values{
		"packagePath": {pkg},
		"hash":        {hash},
	}, nil, nil)
	return err == nil
}

func dashboardPackages(kind string) []string {
	args := url.Values{"kind": []string{kind}}
	var resp []struct {
		Path string
	}
	if err := dash("GET", "packages", args, nil, &resp); err != nil {
		log.Println("dashboardPackages:", err)
		return nil
	}
	if *verbose {
		fmt.Printf("dash resp: %+v\n", resp)
	}
	var pkgs []string
	for _, r := range resp {
		pkgs = append(pkgs, r.Path)
	}
	return pkgs
}
