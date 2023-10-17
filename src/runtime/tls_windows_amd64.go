// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// osSetupTLS is called by needm to set up TLS for non-Go threads.
//
// Defined in assembly.
func osSetupTLS(mp *m)
