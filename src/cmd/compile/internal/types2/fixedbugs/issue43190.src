// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import ; // ERROR missing import path
import
var /* ERROR missing import path */ _ int
import .; // ERROR missing import path

import ()
import (.) // ERROR missing import path
import (
	"fmt"
	.
) // ERROR missing import path

var _ = fmt.Println // avoid imported but not used error
