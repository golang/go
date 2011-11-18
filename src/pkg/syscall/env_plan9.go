// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Plan 9 environment variables.

package syscall

import "errors"

func Getenv(key string) (value string, found bool) {
	if len(key) == 0 {
		return "", false
	}
	f, e := Open("/env/"+key, O_RDONLY)
	if e != nil {
		return "", false
	}
	defer Close(f)

	l, _ := Seek(f, 0, 2)
	Seek(f, 0, 0)
	buf := make([]byte, l)
	n, e := Read(f, buf)
	if e != nil {
		return "", false
	}

	if n > 0 && buf[n-1] == 0 {
		buf = buf[:n-1]
	}
	return string(buf), true
}

func Setenv(key, value string) error {
	if len(key) == 0 {
		return errors.New("bad arg in system call")
	}

	f, e := Create("/env/"+key, O_RDWR, 0666)
	if e != nil {
		return e
	}
	defer Close(f)

	_, e = Write(f, []byte(value))
	return nil
}

func Clearenv() {
	RawSyscall(SYS_RFORK, RFCENVG, 0, 0)
}

func Environ() []string {
	env := make([]string, 0, 100)

	f, e := Open("/env", O_RDONLY)
	if e != nil {
		panic(e)
	}
	defer Close(f)

	names, e := readdirnames(f)
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
