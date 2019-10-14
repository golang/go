// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package auth provides access to user-provided authentication credentials.
package auth

import "net/http"

// AddCredentials fills in the user's credentials for req, if any.
// The return value reports whether any matching credentials were found.
func AddCredentials(req *http.Request) (added bool) {
	// TODO(golang.org/issue/26232): Support arbitrary user-provided credentials.
	netrcOnce.Do(readNetrc)
	for _, l := range netrc {
		if l.machine == req.URL.Host {
			req.SetBasicAuth(l.login, l.password)
			return true
		}
	}

	return false
}
