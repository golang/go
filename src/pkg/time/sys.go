// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import "syscall"

// Sleep pauses the current goroutine for the duration d.
func Sleep(d Duration)

// readFile reads and returns the content of the named file.
// It is a trivial implementation of ioutil.ReadFile, reimplemented
// here to avoid depending on io/ioutil or os.
func readFile(name string) ([]byte, error) {
	f, err := syscall.Open(name, syscall.O_RDONLY, 0)
	if err != nil {
		return nil, err
	}
	defer syscall.Close(f)
	var (
		buf [4096]byte
		ret []byte
		n   int
	)
	for {
		n, err = syscall.Read(f, buf[:])
		if n > 0 {
			ret = append(ret, buf[:n]...)
		}
		if n == 0 || err != nil {
			break
		}
	}
	return ret, err
}
