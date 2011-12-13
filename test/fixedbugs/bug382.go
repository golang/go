// $G $D/$F.dir/pkg.go && $G $D/$F.go || echo "Bug 382"

// Issue 2529

package main
import "./pkg"

var x = pkg.E

var fo = struct {F pkg.T}{F: x}
