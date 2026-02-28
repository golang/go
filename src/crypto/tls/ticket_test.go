// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

var _ = &Config{WrapSession: (&Config{}).EncryptTicket}
var _ = &Config{UnwrapSession: (&Config{}).DecryptTicket}
