// errorcheck

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2598
package foo

return nil // ERROR "non-declaration statement outside function body|expected declaration"
