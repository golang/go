rsc.io/needall 0.0.1
written by hand

-- .mod --
module rsc.io/needall
go 1.23

require rsc.io/needgo121 v0.0.1
require rsc.io/needgo122 v0.0.1
require rsc.io/needgo123 v0.0.1

-- go.mod --
module rsc.io/needall
go 1.23

require rsc.io/needgo121 v0.0.1
require rsc.io/needgo122 v0.0.1
require rsc.io/needgo123 v0.0.1

-- .info --
{"Version":"v0.0.1"}
-- p.go --
package p

func F() {}
