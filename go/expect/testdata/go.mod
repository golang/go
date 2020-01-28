module αfake1α //@mark(αMarker, "αfake1α")

go 1.14

require golang.org/modfile v0.0.0 //@mark(βMarker, "require golang.org/modfile v0.0.0")
//@mark(IndirectMarker, "// indirect")
require golang.org/x/tools v0.0.0-20191219192050-56b0b28a00f7 // indirect