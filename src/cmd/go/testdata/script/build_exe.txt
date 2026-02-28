# go build with -o and -buildmode=exe should report an error on a non-main package.

! go build -buildmode=exe -o out$GOEXE ./not_main
stderr '-buildmode=exe requires exactly one main package'
! exists out$GOEXE
! go build -buildmode=exe -o out$GOEXE ./main_one ./main_two
stderr '-buildmode=exe requires exactly one main package'
! exists out$GOEXE

-- go.mod --
module m

go 1.16
-- not_main/not_main.go --
package not_main

func F() {}
-- main_one/main_one.go --
package main

func main() {}
-- main_two/main_two.go --
package main

func main() {}
