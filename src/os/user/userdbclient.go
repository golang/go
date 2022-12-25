// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package user

// userdbClient queries the io.systemd.UserDatabase VARLINK interface provided by
// systemd-userdbd.service(8) on Linux for obtaining full user/group details
// even when cgo is not available.
// VARLINK protocol: https://varlink.org
// Systemd userdb VARLINK interface https://systemd.io/USER_GROUP_API
// dir contains multiple varlink service sockets implementing the userdb interface.
type userdbClient struct {
	dir string
}

// IsUsable checks if the client can be used to make queries.
func (cl userdbClient) isUsable() bool {
	return len(cl.dir) != 0
}

var defaultUserdbClient userdbClient
