// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Input for TestTypeNamingOrder

// ensures that the order in which "type A B" declarations are
// processed is correct; this was a problem for unified IR imports.

package g

type Client struct {
	common service
	A      *AService
	B      *BService
}

type service struct {
	client *Client
}

type AService service
type BService service
