// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package tracev2 contains definitions for the v2 execution trace wire format.

These definitions are shared between the trace parser and the runtime, so it
must not depend on any package that depends on the runtime (most packages).
*/
package tracev2
