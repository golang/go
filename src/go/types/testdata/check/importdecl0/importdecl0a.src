// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importdecl0

import ()

import (
	// we can have multiple blank imports (was bug)
	_ "math"
	_ "net/rpc"
	init /* ERROR "cannot import package as init" */ "fmt"
	// reflect defines a type "flag" which shows up in the gc export data
	"reflect"
	. /* ERROR "imported but not used" */ "reflect"
)

import "math" /* ERROR "imported but not used" */
import m /* ERROR "imported but not used as m" */ "math"
import _ "math"

import (
	"math/big" /* ERROR "imported but not used" */
	b /* ERROR "imported but not used" */ "math/big"
	_ "math/big"
)

import "fmt"
import f1 "fmt"
import f2 "fmt"

// reflect.flag must not be visible in this package
type flag int
type _ reflect.flag /* ERROR "not exported" */

// imported package name may conflict with local objects
type reflect /* ERROR "reflect already declared" */ int

// dot-imported exported objects may conflict with local objects
type Value /* ERROR "Value already declared through dot-import of package reflect" */ struct{}

var _ = fmt.Println // use "fmt"

func _() {
	f1.Println() // use "fmt"
}

func _() {
	_ = func() {
		f2.Println() // use "fmt"
	}
}
