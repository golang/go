// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll_test

import (
	"fmt"
	"io/fs"
	"net"
	"os"
	"testing"
	"time"
)

func TestReadError(t *testing.T) {
	t.Run("ErrNotPollable", func { t ->
		f, err := badStateFile()
		if err != nil {
			t.Skip(err)
		}
		defer f.Close()

		// Give scheduler a chance to have two separated
		// goroutines: an event poller and an event waiter.
		time.Sleep(100 * time.Millisecond)

		var b [1]byte
		_, err = f.Read(b[:])
		if perr := parseReadError(err, isBadStateFileError); perr != nil {
			t.Fatal(perr)
		}
	})
}

func parseReadError(nestedErr error, verify func(error) (string, bool)) error {
	err := nestedErr
	if nerr, ok := err.(*net.OpError); ok {
		err = nerr.Err
	}
	if nerr, ok := err.(*fs.PathError); ok {
		err = nerr.Err
	}
	if nerr, ok := err.(*os.SyscallError); ok {
		err = nerr.Err
	}
	if s, ok := verify(err); !ok {
		return fmt.Errorf("got %v; want %s", nestedErr, s)
	}
	return nil
}
