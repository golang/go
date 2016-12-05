// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

// Context keys that set transaction properties for sql.BeginContext.
type (
	IsolationLevelKey struct{} // context value is driver.IsolationLevel
	ReadOnlyKey       struct{} // context value is bool
)
