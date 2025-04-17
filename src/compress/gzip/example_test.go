// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gzip_test

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"time"
)

func Example_writerReader() {
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)

	// Setting the Header fields is optional.
	zw.Name = "a-new-hope.txt"
	zw.Comment = "an epic space opera by George Lucas"
	zw.ModTime = time.Date(1977, time.May, 25, 0, 0, 0, 0, time.UTC)

	_, err := zw.Write([]byte("A long time ago in a galaxy far, far away..."))
	if err != nil {
		log.Fatal(err)
	}

	if err := zw.Close(); err != nil {
		log.Fatal(err)
	}

	zr, err := gzip.NewReader(&buf)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Name: %s\nComment: %s\nModTime: %s\n\n", zr.Name, zr.Comment, zr.ModTime.UTC())

	if _, err := io.Copy(os.Stdout, zr); err != nil {
		log.Fatal(err)
	}

	if err := zr.Close(); err != nil {
		log.Fatal(err)
	}

	// Output:
	// Name: a-new-hope.txt
	// Comment: an epic space opera by George Lucas
	// ModTime: 1977-05-25 00:00:00 +0000 UTC
	//
	// A long time ago in a galaxy far, far away...
}

func ExampleReader_Multistream() {
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)

	var files = []struct {
		name    string
		comment string
		modTime time.Time
		data    string
	}{
		{"file-1.txt", "file-header-1", time.Date(2006, time.February, 1, 3, 4, 5, 0, time.UTC), "Hello Gophers - 1"},
		{"file-2.txt", "file-header-2", time.Date(2007, time.March, 2, 4, 5, 6, 1, time.UTC), "Hello Gophers - 2"},
	}

	for _, file := range files {
		zw.Name = file.name
		zw.Comment = file.comment
		zw.ModTime = file.modTime

		if _, err := zw.Write([]byte(file.data)); err != nil {
			log.Fatal(err)
		}

		if err := zw.Close(); err != nil {
			log.Fatal(err)
		}

		zw.Reset(&buf)
	}

	zr, err := gzip.NewReader(&buf)
	if err != nil {
		log.Fatal(err)
	}

	for {
		zr.Multistream(false)
		fmt.Printf("Name: %s\nComment: %s\nModTime: %s\n\n", zr.Name, zr.Comment, zr.ModTime.UTC())

		if _, err := io.Copy(os.Stdout, zr); err != nil {
			log.Fatal(err)
		}

		fmt.Print("\n\n")

		err = zr.Reset(&buf)
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
	}

	if err := zr.Close(); err != nil {
		log.Fatal(err)
	}

	// Output:
	// Name: file-1.txt
	// Comment: file-header-1
	// ModTime: 2006-02-01 03:04:05 +0000 UTC
	//
	// Hello Gophers - 1
	//
	// Name: file-2.txt
	// Comment: file-header-2
	// ModTime: 2007-03-02 04:05:06 +0000 UTC
	//
	// Hello Gophers - 2
}

func Example_compressingReader() {
	// This is an example of writing a compressing reader.
	// This can be useful for an HTTP client body, as shown.

	const testdata = "the data to be compressed"

	// This HTTP handler is just for testing purposes.
	handler := http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		zr, err := gzip.NewReader(req.Body)
		if err != nil {
			log.Fatal(err)
		}

		// Just output the data for the example.
		if _, err := io.Copy(os.Stdout, zr); err != nil {
			log.Fatal(err)
		}
	})
	ts := httptest.NewServer(handler)
	defer ts.Close()

	// The remainder is the example code.

	// The data we want to compress, as an io.Reader
	dataReader := strings.NewReader(testdata)

	// bodyReader is the body of the HTTP request, as an io.Reader.
	// httpWriter is the body of the HTTP request, as an io.Writer.
	bodyReader, httpWriter := io.Pipe()

	// Make sure that bodyReader is always closed, so that the
	// goroutine below will always exit.
	defer bodyReader.Close()

	// gzipWriter compresses data to httpWriter.
	gzipWriter := gzip.NewWriter(httpWriter)

	// errch collects any errors from the writing goroutine.
	errch := make(chan error, 1)

	go func() {
		defer close(errch)
		sentErr := false
		sendErr := func(err error) {
			if !sentErr {
				errch <- err
				sentErr = true
			}
		}

		// Copy our data to gzipWriter, which compresses it to
		// gzipWriter, which feeds it to bodyReader.
		if _, err := io.Copy(gzipWriter, dataReader); err != nil && err != io.ErrClosedPipe {
			sendErr(err)
		}
		if err := gzipWriter.Close(); err != nil && err != io.ErrClosedPipe {
			sendErr(err)
		}
		if err := httpWriter.Close(); err != nil && err != io.ErrClosedPipe {
			sendErr(err)
		}
	}()

	// Send an HTTP request to the test server.
	req, err := http.NewRequest("PUT", ts.URL, bodyReader)
	if err != nil {
		log.Fatal(err)
	}

	// Note that passing req to http.Client.Do promises that it
	// will close the body, in this case bodyReader.
	resp, err := ts.Client().Do(req)
	if err != nil {
		log.Fatal(err)
	}

	// Check whether there was an error compressing the data.
	if err := <-errch; err != nil {
		log.Fatal(err)
	}

	// For this example we don't care about the response.
	resp.Body.Close()

	// Output: the data to be compressed
}
