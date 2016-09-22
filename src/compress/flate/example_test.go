// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate_test

import (
	"bytes"
	"compress/flate"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"sync"
)

// In performance critical applications, Reset can be used to discard the
// current compressor or decompressor state and reinitialize them quickly
// by taking advantage of previously allocated memory.
func Example_reset() {
	proverbs := []string{
		"Don't communicate by sharing memory, share memory by communicating.\n",
		"Concurrency is not parallelism.\n",
		"The bigger the interface, the weaker the abstraction.\n",
		"Documentation is for users.\n",
	}

	var r strings.Reader
	var b bytes.Buffer
	buf := make([]byte, 32<<10)

	zw, err := flate.NewWriter(nil, flate.DefaultCompression)
	if err != nil {
		log.Fatal(err)
	}
	zr := flate.NewReader(nil)

	for _, s := range proverbs {
		r.Reset(s)
		b.Reset()

		// Reset the compressor and encode from some input stream.
		zw.Reset(&b)
		if _, err := io.CopyBuffer(zw, &r, buf); err != nil {
			log.Fatal(err)
		}
		if err := zw.Close(); err != nil {
			log.Fatal(err)
		}

		// Reset the decompressor and decode to some output stream.
		if err := zr.(flate.Resetter).Reset(&b, nil); err != nil {
			log.Fatal(err)
		}
		if _, err := io.CopyBuffer(os.Stdout, zr, buf); err != nil {
			log.Fatal(err)
		}
		if err := zr.Close(); err != nil {
			log.Fatal(err)
		}
	}

	// Output:
	// Don't communicate by sharing memory, share memory by communicating.
	// Concurrency is not parallelism.
	// The bigger the interface, the weaker the abstraction.
	// Documentation is for users.
}

// A preset dictionary can be used to improve the compression ratio.
// The downside to using a dictionary is that the compressor and decompressor
// must agree in advance what dictionary to use.
func Example_dictionary() {
	// The dictionary is a string of bytes. When compressing some input data,
	// the compressor will attempt to substitute substrings with matches found
	// in the dictionary. As such, the dictionary should only contain substrings
	// that are expected to be found in the actual data stream.
	const dict = `<?xml version="1.0"?>` + `<book>` + `<data>` + `<meta name="` + `" content="`

	// The data to compress should (but is not required to) contain frequent
	// substrings that match those in the dictionary.
	const data = `<?xml version="1.0"?>
<book>
	<meta name="title" content="The Go Programming Language"/>
	<meta name="authors" content="Alan Donovan and Brian Kernighan"/>
	<meta name="published" content="2015-10-26"/>
	<meta name="isbn" content="978-0134190440"/>
	<data>...</data>
</book>
`

	var b bytes.Buffer

	// Compress the data using the specially crafted dictionary.
	zw, err := flate.NewWriterDict(&b, flate.DefaultCompression, []byte(dict))
	if err != nil {
		log.Fatal(err)
	}
	if _, err := io.Copy(zw, strings.NewReader(data)); err != nil {
		log.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		log.Fatal(err)
	}

	// The decompressor must use the same dictionary as the compressor.
	// Otherwise, the input may appear as corrupted.
	fmt.Println("Decompressed output using the dictionary:")
	zr := flate.NewReaderDict(bytes.NewReader(b.Bytes()), []byte(dict))
	if _, err := io.Copy(os.Stdout, zr); err != nil {
		log.Fatal(err)
	}
	if err := zr.Close(); err != nil {
		log.Fatal(err)
	}

	fmt.Println()

	// Substitute all of the bytes in the dictionary with a '#' to visually
	// demonstrate the approximate effectiveness of using a preset dictionary.
	fmt.Println("Substrings matched by the dictionary are marked with #:")
	hashDict := []byte(dict)
	for i := range hashDict {
		hashDict[i] = '#'
	}
	zr = flate.NewReaderDict(&b, hashDict)
	if _, err := io.Copy(os.Stdout, zr); err != nil {
		log.Fatal(err)
	}
	if err := zr.Close(); err != nil {
		log.Fatal(err)
	}

	// Output:
	// Decompressed output using the dictionary:
	// <?xml version="1.0"?>
	// <book>
	// 	<meta name="title" content="The Go Programming Language"/>
	// 	<meta name="authors" content="Alan Donovan and Brian Kernighan"/>
	// 	<meta name="published" content="2015-10-26"/>
	// 	<meta name="isbn" content="978-0134190440"/>
	// 	<data>...</data>
	// </book>
	//
	// Substrings matched by the dictionary are marked with #:
	// #####################
	// ######
	// 	############title###########The Go Programming Language"/#
	// 	############authors###########Alan Donovan and Brian Kernighan"/#
	// 	############published###########2015-10-26"/#
	// 	############isbn###########978-0134190440"/#
	// 	######...</#####
	// </#####
}

// DEFLATE is suitable for transmitting compressed data across the network.
func Example_synchronization() {
	var wg sync.WaitGroup
	defer wg.Wait()

	// Use io.Pipe to simulate a network connection.
	// A real network application should take care to properly close the
	// underlying connection.
	rp, wp := io.Pipe()

	// Start a goroutine to act as the transmitter.
	wg.Add(1)
	go func() {
		defer wg.Done()

		zw, err := flate.NewWriter(wp, flate.BestSpeed)
		if err != nil {
			log.Fatal(err)
		}

		b := make([]byte, 256)
		for _, m := range strings.Fields("A long time ago in a galaxy far, far away...") {
			// We use a simple framing format where the first byte is the
			// message length, followed the message itself.
			b[0] = uint8(copy(b[1:], m))

			if _, err := zw.Write(b[:1+len(m)]); err != nil {
				log.Fatal(err)
			}

			// Flush ensures that the receiver can read all data sent so far.
			if err := zw.Flush(); err != nil {
				log.Fatal(err)
			}
		}

		if err := zw.Close(); err != nil {
			log.Fatal(err)
		}
	}()

	// Start a goroutine to act as the receiver.
	wg.Add(1)
	go func() {
		defer wg.Done()

		zr := flate.NewReader(rp)

		b := make([]byte, 256)
		for {
			// Read the message length.
			// This is guaranteed to return for every corresponding
			// Flush and Close on the transmitter side.
			if _, err := io.ReadFull(zr, b[:1]); err != nil {
				if err == io.EOF {
					break // The transmitter closed the stream
				}
				log.Fatal(err)
			}

			// Read the message content.
			n := int(b[0])
			if _, err := io.ReadFull(zr, b[:n]); err != nil {
				log.Fatal(err)
			}

			fmt.Printf("Received %d bytes: %s\n", n, b[:n])
		}
		fmt.Println()

		if err := zr.Close(); err != nil {
			log.Fatal(err)
		}
	}()

	// Output:
	// Received 1 bytes: A
	// Received 4 bytes: long
	// Received 4 bytes: time
	// Received 3 bytes: ago
	// Received 2 bytes: in
	// Received 1 bytes: a
	// Received 6 bytes: galaxy
	// Received 4 bytes: far,
	// Received 3 bytes: far
	// Received 7 bytes: away...
}
