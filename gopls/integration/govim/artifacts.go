// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
)

var bucket = flag.String("bucket", "golang-gopls_integration_tests", "GCS bucket holding test artifacts.")

const usage = `
artifacts [--bucket=<bucket ID>] <cloud build evaluation ID>

Fetch artifacts from an integration test run. Evaluation ID should be extracted
from the cloud build notification.

In order for this to work, the GCS bucket that artifacts were written to must
be publicly readable. By default, this fetches from the
golang-gopls_integration_tests bucket.
`

func main() {
	flag.Usage = func() {
		fmt.Fprint(flag.CommandLine.Output(), usage)
	}
	flag.Parse()
	if flag.NArg() != 1 {
		flag.Usage()
		os.Exit(2)
	}
	evalID := flag.Arg(0)
	logURL := fmt.Sprintf("https://storage.googleapis.com/%s/log-%s.txt", *bucket, evalID)
	if err := download(logURL); err != nil {
		fmt.Fprintf(os.Stderr, "downloading logs: %v", err)
	}
	tarURL := fmt.Sprintf("https://storage.googleapis.com/%s/govim/%s/artifacts.tar.gz", *bucket, evalID)
	if err := download(tarURL); err != nil {
		fmt.Fprintf(os.Stderr, "downloading artifact tarball: %v", err)
	}
}

func download(artifactURL string) error {
	name := path.Base(artifactURL)
	resp, err := http.Get(artifactURL)
	if err != nil {
		return fmt.Errorf("fetching from GCS: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("got status code %d from GCS", resp.StatusCode)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("reading result: %v", err)
	}
	if err := os.WriteFile(name, data, 0644); err != nil {
		return fmt.Errorf("writing artifact: %v", err)
	}
	return nil
}
