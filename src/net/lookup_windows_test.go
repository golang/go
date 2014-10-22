// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bytes"
	"encoding/json"
	"errors"
	"os/exec"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

var nslookupTestServers = []string{"mail.golang.com", "gmail.com"}

func toJson(v interface{}) string {
	data, _ := json.Marshal(v)
	return string(data)
}

func TestLookupMX(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}
	for _, server := range nslookupTestServers {
		mx, err := LookupMX(server)
		if err != nil {
			t.Errorf("failed %s: %s", server, err)
			continue
		}
		if len(mx) == 0 {
			t.Errorf("no results")
			continue
		}
		expected, err := nslookupMX(server)
		if err != nil {
			t.Logf("skipping failed nslookup %s test: %s", server, err)
		}
		sort.Sort(byPrefAndHost(expected))
		sort.Sort(byPrefAndHost(mx))
		if !reflect.DeepEqual(expected, mx) {
			t.Errorf("different results %s:\texp:%v\tgot:%v", server, toJson(expected), toJson(mx))
		}
	}
}

func TestLookupCNAME(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}
	for _, server := range nslookupTestServers {
		cname, err := LookupCNAME(server)
		if err != nil {
			t.Errorf("failed %s: %s", server, err)
			continue
		}
		if cname == "" {
			t.Errorf("no result %s", server)
		}
		expected, err := nslookupCNAME(server)
		if err != nil {
			t.Logf("skipping failed nslookup %s test: %s", server, err)
			continue
		}
		if expected != cname {
			t.Errorf("different results %s:\texp:%v\tgot:%v", server, expected, cname)
		}
	}
}

func TestLookupNS(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}
	for _, server := range nslookupTestServers {
		ns, err := LookupNS(server)
		if err != nil {
			t.Errorf("failed %s: %s", server, err)
			continue
		}
		if len(ns) == 0 {
			t.Errorf("no results")
			continue
		}
		expected, err := nslookupNS(server)
		if err != nil {
			t.Logf("skipping failed nslookup %s test: %s", server, err)
			continue
		}
		sort.Sort(byHost(expected))
		sort.Sort(byHost(ns))
		if !reflect.DeepEqual(expected, ns) {
			t.Errorf("different results %s:\texp:%v\tgot:%v", toJson(server), toJson(expected), ns)
		}
	}
}

func TestLookupTXT(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}
	for _, server := range nslookupTestServers {
		txt, err := LookupTXT(server)
		if err != nil {
			t.Errorf("failed %s: %s", server, err)
			continue
		}
		if len(txt) == 0 {
			t.Errorf("no results")
			continue
		}
		expected, err := nslookupTXT(server)
		if err != nil {
			t.Logf("skipping failed nslookup %s test: %s", server, err)
			continue
		}
		sort.Strings(expected)
		sort.Strings(txt)
		if !reflect.DeepEqual(expected, txt) {
			t.Errorf("different results %s:\texp:%v\tgot:%v", server, toJson(expected), toJson(txt))
		}
	}
}

type byPrefAndHost []*MX

func (s byPrefAndHost) Len() int { return len(s) }
func (s byPrefAndHost) Less(i, j int) bool {
	if s[i].Pref != s[j].Pref {
		return s[i].Pref < s[j].Pref
	}
	return s[i].Host < s[j].Host
}
func (s byPrefAndHost) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

type byHost []*NS

func (s byHost) Len() int           { return len(s) }
func (s byHost) Less(i, j int) bool { return s[i].Host < s[j].Host }
func (s byHost) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

func fqdn(s string) string {
	if len(s) == 0 || s[len(s)-1] != '.' {
		return s + "."
	}
	return s
}

func nslookup(qtype, name string) (string, error) {
	var out bytes.Buffer
	var err bytes.Buffer
	cmd := exec.Command("nslookup", "-querytype="+qtype, name)
	cmd.Stdout = &out
	cmd.Stderr = &err
	if err := cmd.Run(); err != nil {
		return "", err
	}
	r := strings.Replace(out.String(), "\r\n", "\n", -1)
	// nslookup stderr output contains also debug information such as
	// "Non-authoritative answer" and it doesn't return the correct errcode
	if strings.Contains(err.String(), "can't find") {
		return r, errors.New(err.String())
	}
	return r, nil
}

func nslookupMX(name string) (mx []*MX, err error) {
	var r string
	if r, err = nslookup("mx", name); err != nil {
		return
	}
	mx = make([]*MX, 0, 10)
	// linux nslookup syntax
	// golang.org      mail exchanger = 2 alt1.aspmx.l.google.com.
	rx := regexp.MustCompile(`(?m)^([a-z0-9.\-]+)\s+mail exchanger\s*=\s*([0-9]+)\s*([a-z0-9.\-]+)$`)
	for _, ans := range rx.FindAllStringSubmatch(r, -1) {
		pref, _ := strconv.Atoi(ans[2])
		mx = append(mx, &MX{fqdn(ans[3]), uint16(pref)})
	}
	// windows nslookup syntax
	// gmail.com       MX preference = 30, mail exchanger = alt3.gmail-smtp-in.l.google.com
	rx = regexp.MustCompile(`(?m)^([a-z0-9.\-]+)\s+MX preference\s*=\s*([0-9]+)\s*,\s*mail exchanger\s*=\s*([a-z0-9.\-]+)$`)
	for _, ans := range rx.FindAllStringSubmatch(r, -1) {
		pref, _ := strconv.Atoi(ans[2])
		mx = append(mx, &MX{fqdn(ans[3]), uint16(pref)})
	}
	return
}

func nslookupNS(name string) (ns []*NS, err error) {
	var r string
	if r, err = nslookup("ns", name); err != nil {
		return
	}
	ns = make([]*NS, 0, 10)
	// golang.org      nameserver = ns1.google.com.
	rx := regexp.MustCompile(`(?m)^([a-z0-9.\-]+)\s+nameserver\s*=\s*([a-z0-9.\-]+)$`)
	for _, ans := range rx.FindAllStringSubmatch(r, -1) {
		ns = append(ns, &NS{fqdn(ans[2])})
	}
	return
}

func nslookupCNAME(name string) (cname string, err error) {
	var r string
	if r, err = nslookup("cname", name); err != nil {
		return
	}
	// mail.golang.com canonical name = golang.org.
	rx := regexp.MustCompile(`(?m)^([a-z0-9.\-]+)\s+canonical name\s*=\s*([a-z0-9.\-]+)$`)
	// assumes the last CNAME is the correct one
	last := name
	for _, ans := range rx.FindAllStringSubmatch(r, -1) {
		last = ans[2]
	}
	return fqdn(last), nil
}

func nslookupTXT(name string) (txt []string, err error) {
	var r string
	if r, err = nslookup("txt", name); err != nil {
		return
	}
	txt = make([]string, 0, 10)
	// linux
	// golang.org      text = "v=spf1 redirect=_spf.google.com"

	// windows
	// golang.org      text =
	//
	//    "v=spf1 redirect=_spf.google.com"
	rx := regexp.MustCompile(`(?m)^([a-z0-9.\-]+)\s+text\s*=\s*"(.*)"$`)
	for _, ans := range rx.FindAllStringSubmatch(r, -1) {
		txt = append(txt, ans[2])
	}
	return
}
