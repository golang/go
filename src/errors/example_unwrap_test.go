// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package errors_test

import (
	"errors"
	"fmt"
	"os"
	"time"
)

// MyError2 is an error implementation that includes a time, a message, and an
// underlying error.
type MyError2 struct {
	When time.Time
	What string
	err  error
}

func (e MyError2) Error() string {
	return fmt.Sprintf("%v at %v: %v", e.What, e.When, e.err)
}

// Unwrap returns e's underlying error, or nil if there is none.
func (e MyError2) Unwrap() error {
	return e.err
}

func readConfig() error {
	if _, err := os.Open("non-existing"); err != nil {
		return MyError2{
			time.Date(1989, 3, 15, 22, 30, 0, 0, time.UTC),
			"reading config file",
			err,
		}
	}
	return nil
}

func Example_unwrap() {
	if err := readConfig(); err != nil {
		// Display the error.
		fmt.Println(err)
		// If we can retrieve the path, try to recover
		// by taking another action.
		var pe *os.PathError
		if errors.As(err, &pe) {
			restoreFile(pe.Path)
		}
	}
	// Output: reading config file at 1989-03-15 22:30:00 +0000 UTC: open non-existing: no such file or directory
}

func restoreFile(path string) {}
