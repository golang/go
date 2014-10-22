// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that go generate handles command aliases.

//go:generate -command run echo Now is the time
//go:generate run for all good men

package p
