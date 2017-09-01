// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar_test

import (
	"archive/tar"
	"bytes"
	"crypto/md5"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"strings"
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

// A sparse file can efficiently represent a large file that is mostly empty.
// When packing an archive, Header.DetectSparseHoles can be used to populate
// the sparse map, while Header.PunchSparseHoles can be used to create a
// sparse file on disk when extracting an archive.
func Example_sparseAutomatic() {
	// Create the source sparse file.
	src, err := ioutil.TempFile("", "sparse.db")
	if err != nil {
		log.Fatal(err)
	}
	defer os.Remove(src.Name()) // Best-effort cleanup
	defer func() {
		if err := src.Close(); err != nil {
			log.Fatal(err)
		}
	}()
	if err := src.Truncate(10e6); err != nil {
		log.Fatal(err)
	}
	for i := 0; i < 10; i++ {
		if _, err := src.Seek(1e6-1e3, io.SeekCurrent); err != nil {
			log.Fatal(err)
		}
		if _, err := src.Write(bytes.Repeat([]byte{'0' + byte(i)}, 1e3)); err != nil {
			log.Fatal(err)
		}
	}

	// Create an archive and pack the source sparse file to it.
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	fi, err := src.Stat()
	if err != nil {
		log.Fatal(err)
	}
	hdr, err := tar.FileInfoHeader(fi, "")
	if err != nil {
		log.Fatal(err)
	}
	if err := hdr.DetectSparseHoles(src); err != nil {
		log.Fatal(err)
	}
	if err := tw.WriteHeader(hdr); err != nil {
		log.Fatal(err)
	}
	if _, err := io.Copy(tw, src); err != nil {
		log.Fatal(err)
	}
	if err := tw.Close(); err != nil {
		log.Fatal(err)
	}

	// Create the destination sparse file.
	dst, err := ioutil.TempFile("", "sparse.db")
	if err != nil {
		log.Fatal(err)
	}
	defer os.Remove(dst.Name()) // Best-effort cleanup
	defer func() {
		if err := dst.Close(); err != nil {
			log.Fatal(err)
		}
	}()

	// Open the archive and extract the sparse file into the destination file.
	tr := tar.NewReader(&buf)
	hdr, err = tr.Next()
	if err != nil {
		log.Fatal(err)
	}
	if err := hdr.PunchSparseHoles(dst); err != nil {
		log.Fatal(err)
	}
	if _, err := io.Copy(dst, tr); err != nil {
		log.Fatal(err)
	}

	// Verify that the sparse files are identical.
	want, err := ioutil.ReadFile(src.Name())
	if err != nil {
		log.Fatal(err)
	}
	got, err := ioutil.ReadFile(dst.Name())
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Src MD5: %08x\n", md5.Sum(want))
	fmt.Printf("Dst MD5: %08x\n", md5.Sum(got))

	// Output:
	// Src MD5: 33820d648d42cb3da2515da229149f74
	// Dst MD5: 33820d648d42cb3da2515da229149f74
}

// The SparseHoles can be manually constructed without Header.DetectSparseHoles.
func Example_sparseManual() {
	// Define a sparse file to add to the archive.
	// This sparse files contains 5 data fragments, and 4 hole fragments.
	// The logical size of the file is 16 KiB, while the physical size of the
	// file is only 3 KiB (not counting the header data).
	hdr := &tar.Header{
		Name: "sparse.db",
		Size: 16384,
		SparseHoles: []tar.SparseEntry{
			// Data fragment at 0..1023
			{Offset: 1024, Length: 1024 - 512}, // Hole fragment at 1024..1535
			// Data fragment at 1536..2047
			{Offset: 2048, Length: 2048 - 512}, // Hole fragment at 2048..3583
			// Data fragment at 3584..4095
			{Offset: 4096, Length: 4096 - 512}, // Hole fragment at 4096..7679
			// Data fragment at 7680..8191
			{Offset: 8192, Length: 8192 - 512}, // Hole fragment at 8192..15871
			// Data fragment at 15872..16383
		},
	}

	// The regions marked as a sparse hole are filled with NUL-bytes.
	// The total length of the body content must match the specified Size field.
	body := "" +
		strings.Repeat("A", 1024) +
		strings.Repeat("\x00", 1024-512) +
		strings.Repeat("B", 512) +
		strings.Repeat("\x00", 2048-512) +
		strings.Repeat("C", 512) +
		strings.Repeat("\x00", 4096-512) +
		strings.Repeat("D", 512) +
		strings.Repeat("\x00", 8192-512) +
		strings.Repeat("E", 512)

	h := md5.Sum([]byte(body))
	fmt.Printf("Write content of %s, Size: %d, MD5: %08x\n", hdr.Name, len(body), h)
	fmt.Printf("Write SparseHoles of %s:\n\t%v\n\n", hdr.Name, hdr.SparseHoles)

	// Create a new archive and write the sparse file.
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	if err := tw.WriteHeader(hdr); err != nil {
		log.Fatal(err)
	}
	if _, err := tw.Write([]byte(body)); err != nil {
		log.Fatal(err)
	}
	if err := tw.Close(); err != nil {
		log.Fatal(err)
	}

	// Open and iterate through the files in the archive.
	tr := tar.NewReader(&buf)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		body, err := ioutil.ReadAll(tr)
		if err != nil {
			log.Fatal(err)
		}

		h := md5.Sum([]byte(body))
		fmt.Printf("Read content of %s, Size: %d, MD5: %08x\n", hdr.Name, len(body), h)
		fmt.Printf("Read SparseHoles of %s:\n\t%v\n\n", hdr.Name, hdr.SparseHoles)
	}

	// Output:
	// Write content of sparse.db, Size: 16384, MD5: 9b4e2cfae0f9303d30237718e891e9f9
	// Write SparseHoles of sparse.db:
	// 	[{1024 512} {2048 1536} {4096 3584} {8192 7680}]
	//
	// Read content of sparse.db, Size: 16384, MD5: 9b4e2cfae0f9303d30237718e891e9f9
	// Read SparseHoles of sparse.db:
	// 	[{1024 512} {2048 1536} {4096 3584} {8192 7680} {16384 0}]
}
