// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test go generate variable substitution.

//go:generate echo $GOARCH $GOFILE:$GOLINE ${GOPACKAGE}abc xyz$GOPACKAGE/$GOFILE/123

package p
