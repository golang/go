// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that invalid imports are rejected by the compiler.
// Does not compile.

package main

// Each of these pairs tests both `` vs "" strings
// and also use of invalid characters spelled out as
// escape sequences and written directly.
// For example `"\x00"` tests import "\x00"
// while "`\x00`" tests import `<actual-NUL-byte>`.
import ""         // ERROR "import path"
import ``         // ERROR "import path"
import "\x00"     // ERROR "import path"
import `\x00`     // ERROR "import path"
import "\x7f"     // ERROR "import path"
import `\x7f`     // ERROR "import path"
import "a!"       // ERROR "import path"
import `a!`       // ERROR "import path"
import "a b"      // ERROR "import path"
import `a b`      // ERROR "import path"
import "a\\b"     // ERROR "import path"
import `a\\b`     // ERROR "import path"
import "\"`a`\""  // ERROR "import path"
import `\"a\"`    // ERROR "import path"
import "\x80\x80" // ERROR "import path"
import `\x80\x80` // ERROR "import path"
import "\xFFFD"   // ERROR "import path"
import `\xFFFD`   // ERROR "import path"

// Invalid local imports.
import "/foo"  // ERROR "import path cannot be absolute path"
import "c:/foo"  // ERROR "import path contains invalid character"
