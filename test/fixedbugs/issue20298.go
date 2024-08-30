// errorcheck -e=0

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 20298: "imported and not used" error report order was non-deterministic.
// This test works by limiting the number of errors (-e=0)
// and checking that the errors are all at the beginning.

package p

import (
	"bufio"       // ERROR "imported and not used"
	"bytes"       // ERROR "imported and not used"
	"crypto/x509" // ERROR "imported and not used"
	"flag"        // ERROR "imported and not used"
	"fmt"         // ERROR "imported and not used"
	"io"          // ERROR "imported and not used"
	"io/ioutil"   // ERROR "imported and not used"
	"log"         // ERROR "imported and not used"
	"math"        // ERROR "imported and not used"
	"math/big"    // ERROR "imported and not used" "too many errors"
	"math/bits"
	"net"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
)
