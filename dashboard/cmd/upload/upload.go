// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build extdep

// The upload command writes a file to Google Cloud Storage. It's used
// exclusively by the Makefiles in the Go project repos. Think of it
// as a very light version of gsutil or gcloud, but with some
// Go-specific configuration knowledge baked in.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/cloud"
	"google.golang.org/cloud/storage"
)

var (
	public  = flag.Bool("public", false, "object should be world-readable")
	file    = flag.String("file", "-", "Filename to read object from, or '-' for stdin.")
	verbose = flag.Bool("verbose", false, "verbose logging")
)

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: upload [--public] [--file=...] <bucket/object>\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() != 1 {
		flag.Usage()
		os.Exit(1)
	}
	args := strings.SplitN(flag.Arg(0), "/", 2)
	if len(args) != 2 {
		flag.Usage()
		os.Exit(1)
	}
	bucket, object := args[0], args[1]

	proj, ok := bucketProject[bucket]
	if !ok {
		log.Fatalf("bucket %q doesn't have an associated project in upload.go")
	}

	ts, err := tokenSource(bucket)
	if err != nil {
		log.Fatalf("Failed to get an OAuth2 token source: %v", err)
	}
	httpClient := oauth2.NewClient(oauth2.NoContext, ts)

	ctx := cloud.NewContext(proj, httpClient)
	w := storage.NewWriter(ctx, bucket, object)
	// If you don't give the owners access, the web UI seems to
	// have a bug and doesn't have access to see that it's public, so
	// won't render the "Shared Publicly" link. So we do that, even
	// though it's dumb and unnecessary otherwise:
	w.ACL = append(w.ACL, storage.ACLRule{Entity: storage.ACLEntity("project-owners-" + proj), Role: storage.RoleOwner})
	if *public {
		w.ACL = append(w.ACL, storage.ACLRule{Entity: storage.AllUsers, Role: storage.RoleReader})
	}
	var content io.Reader
	if *file == "-" {
		content = os.Stdin
	} else {
		content, err = os.Open(*file)
		if err != nil {
			log.Fatal(err)
		}
	}

	const maxSlurp = 1 << 20
	var buf bytes.Buffer
	n, err := io.CopyN(&buf, content, maxSlurp)
	if err != nil && err != io.EOF {
		log.Fatalf("Error reading from stdin: %v, %v", n, err)
	}
	w.ContentType = http.DetectContentType(buf.Bytes())

	_, err = io.Copy(w, io.MultiReader(&buf, content))
	if cerr := w.Close(); cerr != nil && err == nil {
		err = cerr
	}
	if err != nil {
		log.Fatalf("Write error: %v", err)
	}
	if *verbose {
		log.Printf("Wrote %v", object)
	}
	os.Exit(0)
}

var bucketProject = map[string]string{
	"go-builder-data":       "symbolic-datum-552",
	"http2-demo-server-tls": "symbolic-datum-552",
	"winstrap":              "999119582588",
	"gobuilder":             "999119582588", // deprecated
}

func tokenSource(bucket string) (oauth2.TokenSource, error) {
	proj := bucketProject[bucket]
	fileName := filepath.Join(os.Getenv("HOME"), "keys", proj+".key.json")
	jsonConf, err := ioutil.ReadFile(fileName)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("Missing JSON key configuration. Download the Service Account JSON key from https://console.developers.google.com/project/%s/apiui/credential and place it at %s", proj, fileName)
		}
		return nil, err
	}
	conf, err := google.JWTConfigFromJSON(jsonConf, storage.ScopeReadWrite)
	if err != nil {
		return nil, fmt.Errorf("reading JSON config from %s: %v", fileName, err)
	}
	return conf.TokenSource(oauth2.NoContext), nil
}
