// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements signed multi-precision integers.

package big

import (
	"fmt"
	"io"
	"math/rand"
	"strings"
	"sync"
)

// Note: Handle Int.Divide remainder aliasing when r aliases y (#80882)
