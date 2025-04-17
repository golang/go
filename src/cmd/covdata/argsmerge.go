// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"slices"
	"strconv"
)

type argvalues struct {
	osargs []string
	goos   string
	goarch string
}

type argstate struct {
	state       argvalues
	initialized bool
}

func (a *argstate) Merge(state argvalues) {
	if !a.initialized {
		a.state = state
		a.initialized = true
		return
	}
	if !slices.Equal(a.state.osargs, state.osargs) {
		a.state.osargs = nil
	}
	if state.goos != a.state.goos {
		a.state.goos = ""
	}
	if state.goarch != a.state.goarch {
		a.state.goarch = ""
	}
}

func (a *argstate) ArgsSummary() map[string]string {
	m := make(map[string]string)
	if len(a.state.osargs) != 0 {
		m["argc"] = strconv.Itoa(len(a.state.osargs))
		for k, a := range a.state.osargs {
			m[fmt.Sprintf("argv%d", k)] = a
		}
	}
	if a.state.goos != "" {
		m["GOOS"] = a.state.goos
	}
	if a.state.goarch != "" {
		m["GOARCH"] = a.state.goarch
	}
	return m
}
