// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"internal/testenv"
	"os/exec"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"syscall"
	"testing"
)

var nslookupTestServers = []string{"mail.golang.com", "gmail.com"}
var lookupTestIPs = []string{"8.8.8.8", "1.1.1.1"}

func toJson(v any) string {
	data, _ := json.Marshal(v)
	return string(data)
}

func testLookup(t *testing.T, fn func(*testing.T, *Resolver, string)) {
	for _, def := range []bool{true, false} {
		def := def
		for _, server := range nslookupTestServers {
			server := server
			var name string
			if def {
				name = "default/"
			} else {
				name = "go/"
			}
			t.Run(name+server, func(t *testing.T) {
				t.Parallel()
				r := DefaultResolver
				if !def {
					r = &Resolver{PreferGo: true}
				}
				fn(t, r, server)
			})
		}
	}
}

func TestNSLookupMX(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	testLookup(t, func(t *testing.T, r *Resolver, server string) {
		mx, err := r.LookupMX(context.Background(), server)
		if err != nil {
			t.Fatal(err)
		}
		if len(mx) == 0 {
			t.Fatal("no results")
		}
		expected, err := nslookupMX(server)
		if err != nil {
			t.Skipf("skipping failed nslookup %s test: %s", server, err)
		}
		sort.Sort(byPrefAndHost(expected))
		sort.Sort(byPrefAndHost(mx))
		if !reflect.DeepEqual(expected, mx) {
			t.Errorf("different results %s:\texp:%v\tgot:%v", server, toJson(expected), toJson(mx))
		}
	})
}

func TestNSLookupCNAME(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	testLookup(t, func(t *testing.T, r *Resolver, server string) {
		cname, err := r.LookupCNAME(context.Background(), server)
		if err != nil {
			t.Fatalf("failed %s: %s", server, err)
		}
		if cname == "" {
			t.Fatalf("no result %s", server)
		}
		expected, err := nslookupCNAME(server)
		if err != nil {
			t.Skipf("skipping failed nslookup %s test: %s", server, err)
		}
		if expected != cname {
			t.Errorf("different results %s:\texp:%v\tgot:%v", server, expected, cname)
		}
	})
}

func TestNSLookupNS(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	testLookup(t, func(t *testing.T, r *Resolver, server string) {
		ns, err := r.LookupNS(context.Background(), server)
		if err != nil {
			t.Fatalf("failed %s: %s", server, err)
		}
		if len(ns) == 0 {
			t.Fatal("no results")
		}
		expected, err := nslookupNS(server)
		if err != nil {
			t.Skipf("skipping failed nslookup %s test: %s", server, err)
		}
		sort.Sort(byHost(expected))
		sort.Sort(byHost(ns))
		if !reflect.DeepEqual(expected, ns) {
			t.Errorf("different results %s:\texp:%v\tgot:%v", toJson(server), toJson(expected), ns)
		}
	})
}

func TestNSLookupTXT(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	testLookup(t, func(t *testing.T, r *Resolver, server string) {
		txt, err := r.LookupTXT(context.Background(), server)
		if err != nil {
			t.Fatalf("failed %s: %s", server, err)
		}
		if len(txt) == 0 {
			t.Fatalf("no results")
		}
		expected, err := nslookupTXT(server)
		if err != nil {
			t.Skipf("skipping failed nslookup %s test: %s", server, err)
		}
		sort.Strings(expected)
		sort.Strings(txt)
		if !reflect.DeepEqual(expected, txt) {
			t.Errorf("different results %s:\texp:%v\tgot:%v", server, toJson(expected), toJson(txt))
		}
	})
}

func TestLookupLocalPTR(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	addr, err := localIP()
	if err != nil {
		t.Errorf("failed to get local ip: %s", err)
	}
	names, err := LookupAddr(addr.String())
	if err != nil {
		t.Errorf("failed %s: %s", addr, err)
	}
	if len(names) == 0 {
		t.Errorf("no results")
	}
	expected, err := lookupPTR(addr.String())
	if err != nil {
		t.Skipf("skipping failed lookup %s test: %s", addr.String(), err)
	}
	sort.Strings(expected)
	sort.Strings(names)
	if !reflect.DeepEqual(expected, names) {
		t.Errorf("different results %s:\texp:%v\tgot:%v", addr, toJson(expected), toJson(names))
	}
}

func TestLookupPTR(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	for _, addr := range lookupTestIPs {
		names, err := LookupAddr(addr)
		if err != nil {
			// The DNSError type stores the error as a string, so it cannot wrap the
			// original error code and we cannot check for it here. However, we can at
			// least use its error string to identify the correct localized text for
			// the error to skip.
			var DNS_ERROR_RCODE_SERVER_FAILURE syscall.Errno = 9002
			if strings.HasSuffix(err.Error(), DNS_ERROR_RCODE_SERVER_FAILURE.Error()) {
				testenv.SkipFlaky(t, 38111)
			}
			t.Errorf("failed %s: %s", addr, err)
		}
		if len(names) == 0 {
			t.Errorf("no results")
		}
		expected, err := lookupPTR(addr)
		if err != nil {
			t.Logf("skipping failed lookup %s test: %s", addr, err)
			continue
		}
		sort.Strings(expected)
		sort.Strings(names)
		if !reflect.DeepEqual(expected, names) {
			t.Errorf("different results %s:\texp:%v\tgot:%v", addr, toJson(expected), toJson(names))
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

func nslookup(qtype, name string) (string, error) {
	var out strings.Builder
	var err strings.Builder
	cmd := exec.Command("nslookup", "-querytype="+qtype, name)
	cmd.Stdout = &out
	cmd.Stderr = &err
	if err := cmd.Run(); err != nil {
		return "", err
	}
	r := strings.ReplaceAll(out.String(), "\r\n", "\n")
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
		pref, _, _ := dtoi(ans[2])
		mx = append(mx, &MX{absDomainName(ans[3]), uint16(pref)})
	}
	// windows nslookup syntax
	// gmail.com       MX preference = 30, mail exchanger = alt3.gmail-smtp-in.l.google.com
	rx = regexp.MustCompile(`(?m)^([a-z0-9.\-]+)\s+MX preference\s*=\s*([0-9]+)\s*,\s*mail exchanger\s*=\s*([a-z0-9.\-]+)$`)
	for _, ans := range rx.FindAllStringSubmatch(r, -1) {
		pref, _, _ := dtoi(ans[2])
		mx = append(mx, &MX{absDomainName(ans[3]), uint16(pref)})
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
		ns = append(ns, &NS{absDomainName(ans[2])})
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
	return absDomainName(last), nil
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

func ping(name string) (string, error) {
	cmd := exec.Command("ping", "-n", "1", "-a", name)
	stdoutStderr, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("%v: %v", err, string(stdoutStderr))
	}
	r := strings.ReplaceAll(string(stdoutStderr), "\r\n", "\n")
	return r, nil
}

func lookupPTR(name string) (ptr []string, err error) {
	var r string
	if r, err = ping(name); err != nil {
		return
	}
	ptr = make([]string, 0, 10)
	rx := regexp.MustCompile(`(?m)^Pinging\s+([a-zA-Z0-9.\-]+)\s+\[.*$`)
	for _, ans := range rx.FindAllStringSubmatch(r, -1) {
		ptr = append(ptr, absDomainName(ans[1]))
	}
	return
}

func localIP() (ip IP, err error) {
	conn, err := Dial("udp", "golang.org:80")
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	localAddr := conn.LocalAddr().(*UDPAddr)

	return localAddr.IP, nil
}
