// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import "net/http"

func clientPriorityDisabled(s *http.Server) bool {
	return s.DisableClientPriority
}
