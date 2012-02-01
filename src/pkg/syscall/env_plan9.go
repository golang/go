// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Plan 9 environment variables.

package syscall

import (
	"errors"
	"sync"
)

var (
	// envOnce guards initialization by copyenv, which populates env.
	envOnce sync.Once

	// envLock guards env.
	envLock sync.RWMutex

	// env maps from an environment variable to its value.
	env map[string]string
)

func readenv(key string) (string, error) {
	fd, err := Open("/env/"+key, O_RDONLY)
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
	fd, err := Create("/env/"+key, O_RDWR, 0666)
	if err != nil {
		return err
	}
	defer Close(fd)
	_, err = Write(fd, []byte(value))
	return err
}

func copyenv() {
	env = make(map[string]string)
	fd, err := Open("/env", O_RDONLY)
	if err != nil {
		return
	}
	defer Close(fd)
	files, err := readdirnames(fd)
	if err != nil {
		return
	}
	for _, key := range files {
		v, err := readenv(key)
		if err != nil {
			continue
		}
		env[key] = v
	}
}

func Getenv(key string) (value string, found bool) {
	envOnce.Do(copyenv)
	if len(key) == 0 {
		return "", false
	}

	envLock.RLock()
	defer envLock.RUnlock()

	v, ok := env[key]
	if !ok {
		return "", false
	}
	return v, true
}

func Setenv(key, value string) error {
	envOnce.Do(copyenv)
	if len(key) == 0 {
		return errors.New("zero length key")
	}

	envLock.Lock()
	defer envLock.Unlock()

	err := writeenv(key, value)
	if err != nil {
		return err
	}
	env[key] = value
	return nil
}

func Clearenv() {
	envOnce.Do(copyenv) // prevent copyenv in Getenv/Setenv

	envLock.Lock()
	defer envLock.Unlock()

	env = make(map[string]string)
	RawSyscall(SYS_RFORK, RFCENVG, 0, 0)
}

func Environ() []string {
	envOnce.Do(copyenv)
	envLock.RLock()
	defer envLock.RUnlock()
	a := make([]string, len(env))
	i := 0
	for k, v := range env {
		a[i] = k + "=" + v
		i++
	}
	return a
}
