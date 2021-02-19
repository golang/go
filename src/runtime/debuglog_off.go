// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !debuglog
// +build !debuglog

package runtime

const dlogEnabled = false

type dlogPerM struct{}

func getCachedDlogger() *dlogger {
	return nil
}

func putCachedDlogger(l *dlogger) bool {
	return false
}
