// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !linux

package user

import "context"

func (cl userdbClient) lookupGroup(_ context.Context, _ string) (*Group, bool, error) {
	return nil, false, nil
}

func (cl userdbClient) lookupGroupId(_ context.Context, _ string) (*Group, bool, error) {
	return nil, false, nil
}

func (cl userdbClient) lookupUser(_ context.Context, _ string) (*User, bool, error) {
	return nil, false, nil
}

func (cl userdbClient) lookupUserId(_ context.Context, _ string) (*User, bool, error) {
	return nil, false, nil
}

func (cl userdbClient) lookupGroupIds(_ context.Context, _ string) ([]string, bool, error) {
	return nil, false, nil
}
