// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"http"
	"json"
	"log"
	"os"
	"regexp"
	"strconv"
)

// getHighWater returns the current highwater revision hash for this builder
func (b *Builder) getHighWater() (rev string, err os.Error) {
	url := fmt.Sprintf("http://%s/hw-get?builder=%s", *dashboard, b.name)
	r, _, err := http.Get(url)
	if err != nil {
		return
	}
	buf := new(bytes.Buffer)
	_, err = buf.ReadFrom(r.Body)
	if err != nil {
		return
	}
	r.Body.Close()
	return buf.String(), nil
}

// recordResult sends build results to the dashboard
func (b *Builder) recordResult(buildLog string, c Commit) os.Error {
	return httpCommand("build", map[string]string{
		"builder": b.name,
		"key":     b.key,
		"node":    c.node,
		"parent":  c.parent,
		"user":    c.user,
		"date":    c.date,
		"desc":    c.desc,
		"log":     buildLog,
	})
}

// match lines like: "package.BechmarkFunc	100000	    999 ns/op"
var benchmarkRegexp = regexp.MustCompile("([^\n\t ]+)[\t ]+([0-9]+)[\t ]+([0-9]+) ns/op")

// recordBenchmarks sends benchmark results to the dashboard
func (b *Builder) recordBenchmarks(benchLog string, c Commit) os.Error {
	results := benchmarkRegexp.FindAllStringSubmatch(benchLog, -1)
	var buf bytes.Buffer
	b64 := base64.NewEncoder(base64.StdEncoding, &buf)
	for _, r := range results {
		for _, s := range r[1:] {
			binary.Write(b64, binary.BigEndian, uint16(len(s)))
			b64.Write([]byte(s))
		}
	}
	b64.Close()
	return httpCommand("benchmarks", map[string]string{
		"builder":       b.name,
		"key":           b.key,
		"node":          c.node,
		"benchmarkdata": buf.String(),
	})
}

// getPackages fetches a list of package paths from the dashboard
func getPackages() (pkgs []string, err os.Error) {
	r, _, err := http.Get(fmt.Sprintf("http://%v/package?fmt=json", *dashboard))
	if err != nil {
		return
	}
	defer r.Body.Close()
	d := json.NewDecoder(r.Body)
	var resp struct {
		Packages []struct {
			Path string
		}
	}
	if err = d.Decode(&resp); err != nil {
		return
	}
	for _, p := range resp.Packages {
		pkgs = append(pkgs, p.Path)
	}
	return
}

// updatePackage sends package build results and info to the dashboard
func (b *Builder) updatePackage(pkg string, state bool, buildLog, info string, c Commit) os.Error {
	args := map[string]string{
		"builder": b.name,
		"key":     b.key,
		"path":    pkg,
		"state":   strconv.Btoa(state),
		"log":     buildLog,
		"info":    info,
		"go_rev":  strconv.Itoa(c.num),
	}
	return httpCommand("package", args)
}

func httpCommand(cmd string, args map[string]string) os.Error {
	if *verbose {
		log.Println("httpCommand", cmd, args)
	}
	url := fmt.Sprintf("http://%v/%v", *dashboard, cmd)
	_, err := http.PostForm(url, args)
	return err
}
