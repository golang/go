// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux openbsd

// Unix environment variables.

package syscall

import "sync"

var env map[string]string
var envOnce sync.Once
var envs []string // provided by runtime

func setenv_c(k, v string)

func copyenv() {
	env = make(map[string]string)
	for _, s := range envs {
		for j := 0; j < len(s); j++ {
			if s[j] == '=' {
				env[s[0:j]] = s[j+1:]
				break
			}
		}
	}
}

var envLock sync.RWMutex

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
		return EINVAL
	}

	envLock.Lock()
	defer envLock.Unlock()

	env[key] = value
	setenv_c(key, value) // is a no-op if cgo isn't loaded
	return nil
}

func Clearenv() {
	envOnce.Do(copyenv) // prevent copyenv in Getenv/Setenv

	envLock.Lock()
	defer envLock.Unlock()

	env = make(map[string]string)

	// TODO(bradfitz): pass through to C
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
