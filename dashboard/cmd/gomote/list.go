// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build extdep

package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"golang.org/x/tools/dashboard/buildlet"
)

func list(args []string) error {
	fs := flag.NewFlagSet("list", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "list usage: gomote list\n\n")
		fs.PrintDefaults()
		os.Exit(1)
	}
	fs.Parse(args)
	if fs.NArg() != 0 {
		fs.Usage()
	}

	prefix := fmt.Sprintf("mote-%s-", username())
	vms, err := buildlet.ListVMs(projTokenSource(), *proj, *zone)
	if err != nil {
		return fmt.Errorf("failed to list VMs: %v", err)
	}
	for _, vm := range vms {
		if !strings.HasPrefix(vm.Name, prefix) {
			continue
		}
		fmt.Printf("%s\thttps://%s\n", vm.Type, strings.TrimSuffix(vm.IPPort, ":443"))
	}
	return nil
}

func namedClient(name string) (*buildlet.Client, error) {
	// TODO(bradfitz): cache the list on disk and avoid the API call?
	vms, err := buildlet.ListVMs(projTokenSource(), *proj, *zone)
	if err != nil {
		return nil, fmt.Errorf("error listing VMs while looking up %q: %v", name, err)
	}
	wantName := fmt.Sprintf("mote-%s-%s", username(), name)
	for _, vm := range vms {
		if vm.Name == wantName {
			return buildlet.NewClient(vm.IPPort, vm.TLS), nil
		}
	}
	return nil, fmt.Errorf("buildlet %q not running", name)
}
