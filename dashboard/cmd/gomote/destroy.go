// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build extdep

package main

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"golang.org/x/tools/dashboard/buildlet"
)

func destroy(args []string) error {
	fs := flag.NewFlagSet("destroy", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "create usage: gomote destroy <instance>\n\n")
		fs.PrintDefaults()
		os.Exit(1)
	}

	fs.Parse(args)
	if fs.NArg() != 1 {
		fs.Usage()
	}
	name := fs.Arg(0)
	bc, err := namedClient(name)
	if err != nil {
		return err
	}

	// Ask it to kill itself, and tell GCE to kill it too:
	gceErrc := make(chan error, 1)
	buildletErrc := make(chan error, 1)
	go func() {
		gceErrc <- buildlet.DestroyVM(projTokenSource(), *proj, *zone, fmt.Sprintf("mote-%s-%s", username(), name))
	}()
	go func() {
		buildletErrc <- bc.Destroy()
	}()
	timeout := time.NewTimer(5 * time.Second)
	defer timeout.Stop()

	var retErr error
	var gceDone, buildletDone bool
	for !gceDone || !buildletDone {
		select {
		case err := <-gceErrc:
			if err != nil {
				log.Printf("GCE: %v", err)
				retErr = err
			} else {
				log.Printf("Requested GCE delete.")
			}
			gceDone = true
		case err := <-buildletErrc:
			if err != nil {
				log.Printf("Buildlet: %v", err)
				retErr = err
			} else {
				log.Printf("Requested buildlet to shut down.")
			}
			buildletDone = true
		case <-timeout.C:
			if !buildletDone {
				log.Printf("timeout asking buildlet to shut down")
			}
			if !gceDone {
				log.Printf("timeout asking GCE to delete builder VM")
			}
			return errors.New("timeout")
		}
	}
	return retErr
}
