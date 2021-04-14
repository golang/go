// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar_test

import (
	"archive/tar"
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
)

func Example_minimal() {
	// Create and add some files to the archive.
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	var files = []struct {
		Name, Body string
	}{
		{"readme.txt", "This archive contains some text files."},
		{"gopher.txt", "Gopher names:\nGeorge\nGeoffrey\nGonzo"},
		{"todo.txt", "Get animal handling license."},
	}
	for _, file := range files {
		hdr := &tar.Header{
			Name: file.Name,
			Mode: 0600,
			Size: int64(len(file.Body)),
		}
		if err := tw.WriteHeader(hdr); err != nil {
			log.Fatal(err)
		}
		if _, err := tw.Write([]byte(file.Body)); err != nil {
			log.Fatal(err)
		}
	}
	if err := tw.Close(); err != nil {
		log.Fatal(err)
	}

	// Open and iterate through the files in the archive.
	tr := tar.NewReader(&buf)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break // End of archive
		}
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Contents of %s:\n", hdr.Name)
		if _, err := io.Copy(os.Stdout, tr); err != nil {
			log.Fatal(err)
		}
		fmt.Println()
	}

	// Output:
	// Contents of readme.txt:
	// This archive contains some text files.
	// Contents of gopher.txt:
	// Gopher names:
	// George
	// Geoffrey
	// Gonzo
	// Contents of todo.txt:
	// Get animal handling license.
}
