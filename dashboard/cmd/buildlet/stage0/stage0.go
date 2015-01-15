// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build stage0

// The stage0 command looks up the buildlet's URL from the GCE metadata
// service, downloads it, and runs it. It's used primarily by Windows,
// since it can be written in a couple lines of shell elsewhere.
package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"google.golang.org/cloud/compute/metadata"
)

const attr = "buildlet-binary-url"

func main() {
	buildletURL, err := metadata.InstanceAttributeValue(attr)
	if err != nil {
		sleepFatalf("Failed to look up %q attribute value: %v", attr, err)
	}
	target := filepath.FromSlash("./buildlet.exe")
	if err := download(target, buildletURL); err != nil {
		sleepFatalf("Downloading %s: %v", buildletURL, err)
	}
	cmd := exec.Command(target)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		sleepFatalf("Error running buildlet: %v", err)
	}
}

func sleepFatalf(format string, args ...interface{}) {
	log.Printf(format, args...)
	time.Sleep(time.Minute) // so user has time to see it in cmd.exe, maybe
	os.Exit(1)
}

func download(file, url string) error {
	log.Printf("Downloading %s to %s ...\n", url, file)
	res, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("Error fetching %v: %v", url, err)
	}
	if res.StatusCode != 200 {
		return fmt.Errorf("HTTP status code of %s was %v", url, res.Status)
	}
	tmp := file + ".tmp"
	os.Remove(tmp)
	os.Remove(file)
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	n, err := io.Copy(f, res.Body)
	res.Body.Close()
	if err != nil {
		return fmt.Errorf("Error reading %v: %v", url, err)
	}
	f.Close()
	err = os.Rename(tmp, file)
	if err != nil {
		return err
	}
	log.Printf("Downloaded %s (%d bytes)", file, n)
	return nil
}
