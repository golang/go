// errorcheckandrundir -1

//go:build !nacl && !js && !plan9

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ignored

// Compile: bug0.go, bug1.go
// Compile and errorCheck: bug2.go
// Link and run: bug3.go
