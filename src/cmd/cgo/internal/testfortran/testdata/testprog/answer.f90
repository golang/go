! Copyright 2016 The Go Authors. All rights reserved.
! Use of this source code is governed by a BSD-style
! license that can be found in the LICENSE file.

function the_answer() result(j) bind(C)
  use iso_c_binding, only: c_int
  integer(c_int) :: j ! output
  j = 42
end function the_answer
