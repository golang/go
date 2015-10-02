// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Plan 9 environment variables.

package syscall

import (
	"errors"
)

var (
	errZeroLengthKey = errors.New("zero length key")
	errShortWrite    = errors.New("i/o count too small")
)

func readenv(key string) (string, error) {
	fd, err := open("/env/"+key, O_RDONLY)
	if err != nil {
		return "", err
	}
	defer Close(fd)
	l, _ := Seek(fd, 0, 2)
	Seek(fd, 0, 0)
	buf := make([]byte, l)
	n, err := Read(fd, buf)
	if err != nil {
		return "", err
	}
	if n > 0 && buf[n-1] == 0 {
		buf = buf[:n-1]
	}
	return string(buf), nil
}

func writeenv(key, value string) error {
	fd, err := create("/env/"+key, O_RDWR, 0666)
	if err != nil {
		return err
	}
	defer Close(fd)
	b := []byte(value)
	n, err := Write(fd, b)
	if err != nil {
		return err
	}
	if n != len(b) {
		return errShortWrite
	}
	return nil
}

func Getenv(key string) (value string, found bool) {
	if len(key) == 0 {
		return "", false
	}
	v, err := readenv(key)
	if err != nil {
		return "", false
	}
	return v, true
}

func Setenv(key, value string) error {
	if len(key) == 0 {
		return errZeroLengthKey
	}
	err := writeenv(key, value)
	if err != nil {
		return err
	}
	return nil
}

func Clearenv() {
	RawSyscall(SYS_RFORK, RFCENVG, 0, 0)
}

func Unsetenv(key string) error {
	if len(key) == 0 {
		return errZeroLengthKey
	}
	Remove("/env/" + key)
	return nil
}

func Environ() []string {
	fd, err := open("/env", O_RDONLY)
	if err != nil {
		return nil
	}
	defer Close(fd)
	files, err := readdirnames(fd)
	if err != nil {
		return nil
	}
	ret := make([]string, 0, len(files))

	for _, key := range files {
		v, err := readenv(key)
		if err != nil {
			continue
		}
		ret = append(ret, key+"="+v)
	}
	return ret
}
