// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Plan 9 environment variables.

package syscall

import "errors"

func Getenv(key string) (value string, found bool) {
	if len(key) == 0 {
		return "", EINVAL
	}
	f, e := Open("/env/" + key)
	if e != nil {
		return "", ENOENV
	}
	defer f.Close()

	l, _ := f.Seek(0, 2)
	f.Seek(0, 0)
	buf := make([]byte, l)
	n, e := f.Read(buf)
	if e != nil {
		return "", ENOENV
	}

	if n > 0 && buf[n-1] == 0 {
		buf = buf[:n-1]
	}
	return string(buf), nil
}

func Setenv(key, value string) error {
	if len(key) == 0 {
		return EINVAL
	}

	f, e := Create("/env/" + key)
	if e != nil {
		return e
	}
	defer f.Close()

	_, e = f.Write([]byte(value))
	return nil
}

func Clearenv() {
	RawSyscall(SYS_RFORK, RFCENVG, 0, 0)
}

func Environ() []string {
	env := make([]string, 0, 100)

	f, e := Open("/env")
	if e != nil {
		panic(e)
	}
	defer f.Close()

	names, e := f.Readdirnames(-1)
	if e != nil {
		panic(e)
	}

	for _, k := range names {
		if v, ok := Getenv(k); ok {
			env = append(env, k+"="+v)
		}
	}
	return env[0:len(env)]
}
