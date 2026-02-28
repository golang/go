// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _(... /* ERROR [.][.][.] is missing type */ )
func _(... /* ERROR [.][.][.] is missing type */ , int)

func _(a, b ... /* ERROR [.][.][.] is missing type */ )
func _(a, b ... /* ERROR [.][.][.] is missing type */ , x int)

func _()(... /* ERROR [.][.][.] is missing type */ )
