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

type obj map[string]interface{}

// dash runs the given method and command on the dashboard.
// If args is non-nil it is encoded as the URL query string.
// If req is non-nil it is JSON-encoded and passed as the body of the HTTP POST.
// If resp is non-nil the server's response is decoded into the value pointed
// to by resp (resp must be a pointer).
func dash(meth, cmd string, args url.Values, req, resp interface{}) error {
	var r *http.Response
	var err error
	if *verbose {
		log.Println("dash", meth, cmd, args, req)
	}
	cmd = "http://" + *dashboard + "/" + cmd
	if len(args) > 0 {
		cmd += "?" + args.Encode()
	}
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
	if result.Error != "" {
		return errors.New(result.Error)
	}

	return nil
}

// todo returns the next hash to build.
func (b *Builder) todo(pkg, goHash string) (rev string, err error) {
	args := url.Values{
		"builder":     {b.name},
		"packagePath": {pkg},
		"goHash":      {goHash},
	}
	var resp string
	if err = dash("GET", "todo", args, nil, &resp); err != nil {
		return
	}
	if resp != "" {
		rev = resp
	}
	return
}

// recordResult sends build results to the dashboard
func (b *Builder) recordResult(ok bool, pkg, hash, goHash, buildLog string) error {
	req := obj{
		"Builder":     b.name,
		"PackagePath": pkg,
		"Hash":        hash,
		"GoHash":      goHash,
		"OK":          ok,
		"Log":         buildLog,
	}
	return dash("POST", "result", url.Values{"key": {b.key}}, req, nil)
}

// packages fetches a list of package paths from the dashboard
func packages() (pkgs []string, err error) {
	return nil, nil
	/* TODO(adg): un-stub this once the new package builder design is done
	var resp struct {
		Packages []struct {
			Path string
		}
	}
	err = dash("GET", "package", &resp, param{"fmt": "json"})
	if err != nil {
		return
	}
	for _, p := range resp.Packages {
		pkgs = append(pkgs, p.Path)
	}
	return
	*/
}

// updatePackage sends package build results and info dashboard
func (b *Builder) updatePackage(pkg string, ok bool, buildLog, info string) error {
	return nil
	/* TODO(adg): un-stub this once the new package builder design is done
	return dash("POST", "package", nil, param{
		"builder": b.name,
		"key":     b.key,
		"path":    pkg,
		"ok":      strconv.FormatBool(ok),
		"log":     buildLog,
		"info":    info,
	})
	*/
}

func postCommit(key, pkg string, l *HgLog) error {
	t, err := time.Parse(time.RFC3339, l.Date)
	if err != nil {
		return fmt.Errorf("parsing %q: %v", l.Date, t)
	}
	return dash("POST", "commit", url.Values{"key": {key}}, obj{
		"PackagePath": pkg,
		"Hash":        l.Hash,
		"ParentHash":  l.Parent,
		"Time":        t.Unix() * 1e6, // in microseconds, yuck!
		"User":        l.Author,
		"Desc":        l.Desc,
	}, nil)
}

func dashboardCommit(pkg, hash string) bool {
	err := dash("GET", "commit", url.Values{
		"packagePath": {pkg},
		"hash":        {hash},
	}, nil, nil)
	return err == nil
}

func dashboardPackages() []string {
	var resp []struct {
		Path string
	}
	if err := dash("GET", "packages", nil, nil, &resp); err != nil {
		log.Println("dashboardPackages:", err)
		return nil
	}
	var pkgs []string
	for _, r := range resp {
		pkgs = append(pkgs, r.Path)
	}
	return pkgs
}
