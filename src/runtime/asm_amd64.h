// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Define features that are guaranteed to be supported by setting the AMD64 variable.
// If a feature is supported, there's no need to check it at runtime every time.

#ifdef GOAMD64_v3
#define hasAVX2
#endif

#ifdef GOAMD64_v4
#define hasAVX2
#endif
