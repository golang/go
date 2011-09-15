// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux openbsd

// Unix environment variables.

package os

import (
	"sync"
)

// ENOENV is the Error indicating that an environment variable does not exist.
var ENOENV = NewError("no such environment variable")

var env map[string]string
var once sync.Once

func copyenv() {
	env = make(map[string]string)
	for _, s := range Envs {
		for j := 0; j < len(s); j++ {
			if s[j] == '=' {
				env[s[0:j]] = s[j+1:]
				break
			}
		}
	}
}

var envLock sync.RWMutex

// Getenverror retrieves the value of the environment variable named by the key.
// It returns the value and an error, if any.
func Getenverror(key string) (value string, err Error) {
	once.Do(copyenv)

	if len(key) == 0 {
		return "", EINVAL
	}

	envLock.RLock()
	defer envLock.RUnlock()

	v, ok := env[key]
	if !ok {
		return "", ENOENV
	}
	return v, nil
}

// Getenv retrieves the value of the environment variable named by the key.
// It returns the value, which will be empty if the variable is not present.
func Getenv(key string) string {
	v, _ := Getenverror(key)
	return v
}

// Setenv sets the value of the environment variable named by the key.
// It returns an Error, if any.
func Setenv(key, value string) Error {
	once.Do(copyenv)
	if len(key) == 0 {
		return EINVAL
	}

	envLock.Lock()
	defer envLock.Unlock()

	env[key] = value
	setenv_c(key, value) // is a no-op if cgo isn't loaded
	return nil
}

// Clearenv deletes all environment variables.
func Clearenv() {
	once.Do(copyenv) // prevent copyenv in Getenv/Setenv

	envLock.Lock()
	defer envLock.Unlock()

	env = make(map[string]string)

	// TODO(bradfitz): pass through to C
}

// Environ returns an array of strings representing the environment,
// in the form "key=value".
func Environ() []string {
	once.Do(copyenv)
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

// TempDir returns the default directory to use for temporary files.
func TempDir() string {
	dir := Getenv("TMPDIR")
	if dir == "" {
		dir = "/tmp"
	}
	return dir
}
