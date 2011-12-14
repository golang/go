// $G $D/$F.go || echo "Bug387"

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2549

/*  Used to die with
missing typecheck: [7f5bf07b4438]

.   AS l(45)
.   .   NAME-main.autotmp_0017 u(1) a(1) l(45) x(0+0) class(PAUTO)
esc(N) tc(1) used(1) ARRAY-[2]string
internal compiler error: missing typecheck 
*/
package main

import (
        "fmt"
        "path/filepath"
)

func main() {
        switch _, err := filepath.Glob(filepath.Join(".", "vnc")); {
        case err != nil:
                fmt.Println(err)
        }
}

