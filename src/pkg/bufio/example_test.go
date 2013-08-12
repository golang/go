// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bufio_test

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func ExampleWriter() {
	w := bufio.NewWriter(os.Stdout)
	fmt.Fprint(w, "Hello, ")
	fmt.Fprint(w, "world!")
	w.Flush() // Don't forget to flush!
	// Output: Hello, world!
}

// The simplest use of a Scanner, to read standard input as a set of lines.
func ExampleScanner_lines() {
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		fmt.Println(scanner.Text()) // Println will add back the final '\n'
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "reading standard input:", err)
	}
}

// Use a Scanner to implement a simple word-count utility by scanning the
// input as a sequence of space-delimited tokens.
func ExampleScanner_words() {
	// An artificial input source.
	const input = "Now is the winter of our discontent,\nMade glorious summer by this sun of York.\n"
	scanner := bufio.NewScanner(strings.NewReader(input))
	// Set the split function for the scanning operation.
	scanner.Split(bufio.ScanWords)
	// Count the words.
	count := 0
	for scanner.Scan() {
		count++
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "reading input:", err)
	}
	fmt.Printf("%d\n", count)
	// Output: 15
}

// Use a Scanner with a custom split function (built by wrapping ScanWords) to validate
// 32-bit decimal input.
func ExampleScanner_custom() {
	// An artificial input source.
	const input = "1234 5678 1234567901234567890"
	scanner := bufio.NewScanner(strings.NewReader(input))
	// Create a custom split function by wrapping the existing ScanWords function.
	split := func(data []byte, atEOF bool) (advance int, token []byte, err error) {
		advance, token, err = bufio.ScanWords(data, atEOF)
		if err == nil && token != nil {
			_, err = strconv.ParseInt(string(token), 10, 32)
		}
		return
	}
	// Set the split function for the scanning operation.
	scanner.Split(split)
	// Validate the input
	for scanner.Scan() {
		fmt.Printf("%s\n", scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("Invalid input: %s", err)
	}
	// Output:
	// 1234
	// 5678
	// Invalid input: strconv.ParseInt: parsing "1234567901234567890": value out of range
}
