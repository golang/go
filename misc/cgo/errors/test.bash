# Copyright 2013 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

if go tool cgo err1.go >errs 2>&1; then
  echo 1>&2 misc/cgo/errors/test.bash: BUG: expected cgo to fail but it succeeded
  exit 1
fi
if ! test -s errs; then
  echo 1>&2 misc/cgo/errors/test.bash: BUG: expected error output but saw none
  exit 1
fi
if ! fgrep err1.go:7 errs >/dev/null 2>&1; then
  echo 1>&2 misc/cgo/errors/test.bash: BUG: expected error on line 7 but saw:
  cat 1>&2 errs
  exit 1
fi
rm -rf errs _obj
exit 0
