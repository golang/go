// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build extdep

package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
	"time"

	"golang.org/x/tools/dashboard"
	"golang.org/x/tools/dashboard/buildlet"
)

func create(args []string) error {
	fs := flag.NewFlagSet("create", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "create usage: gomote create [create-opts] <type>\n\n")
		fs.PrintDefaults()
		os.Exit(1)
	}
	var timeout time.Duration
	fs.DurationVar(&timeout, "timeout", 60*time.Minute, "how long the VM will live before being deleted.")

	fs.Parse(args)
	if fs.NArg() != 1 {
		fs.Usage()
	}
	builderType := fs.Arg(0)
	conf, ok := dashboard.Builders[builderType]
	if !ok || !conf.UsesVM() {
		var valid []string
		var prefixMatch []string
		for k, conf := range dashboard.Builders {
			if conf.UsesVM() {
				valid = append(valid, k)
				if strings.HasPrefix(k, builderType) {
					prefixMatch = append(prefixMatch, k)
				}
			}
		}
		if len(prefixMatch) == 1 {
			builderType = prefixMatch[0]
			conf, _ = dashboard.Builders[builderType]
		} else {
			sort.Strings(valid)
			return fmt.Errorf("Invalid builder type %q. Valid options include: %q", builderType, valid)
		}
	}

	instName := fmt.Sprintf("mote-%s-%s", username(), builderType)
	client, err := buildlet.StartNewVM(projTokenSource(), instName, builderType, buildlet.VMOpts{
		Zone:        *zone,
		ProjectID:   *proj,
		TLS:         userKeyPair(),
		DeleteIn:    timeout,
		Description: fmt.Sprintf("gomote buildlet for %s", username()),
		OnInstanceRequested: func() {
			log.Printf("Sent create request. Waiting for operation.")
		},
		OnInstanceCreated: func() {
			log.Printf("Instance created.")
		},
	})
	if err != nil {
		return fmt.Errorf("failed to create VM: %v", err)
	}
	fmt.Printf("%s\t%s\n", builderType, client.URL())
	return nil
}
