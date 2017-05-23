// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package race contains helper functions for manually instrumenting code for the race detector.

The runtime package intentionally exports these functions only in the race build;
this package exports them unconditionally but without the "race" build tag they are no-ops.
*/
package race
