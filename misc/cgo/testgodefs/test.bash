# Copyright 2014 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# We are testing cgo -godefs, which translates Go files that use
# import "C" into Go files with Go definitions of types defined in the
# import "C" block.  Add more tests here.
FILE_PREFIXES="anonunion issue8478"

RM=
for FP in $FILE_PREFIXES
do
  go tool cgo -godefs ${FP}.go > ${FP}_defs.go
  RM="${RM} ${FP}_defs.go"
done

go build . && ./testgodefs
EXIT=$?
rm -rf _obj testgodefs ${RM}
exit $EXIT
