// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT ·boolVal(SB), NOSPLIT, $0
  CallImport
  RET

TEXT ·intVal(SB), NOSPLIT, $0
  CallImport
  RET

TEXT ·floatVal(SB), NOSPLIT, $0
  CallImport
  RET

TEXT ·stringVal(SB), NOSPLIT, $0
  CallImport
  RET

TEXT ·Value·Get(SB), NOSPLIT, $0
  CallImport
  RET

TEXT ·Value·set(SB), NOSPLIT, $0
  CallImport
  RET

TEXT ·Value·Index(SB), NOSPLIT, $0
  CallImport
  RET

TEXT ·Value·setIndex(SB), NOSPLIT, $0
  CallImport
  RET

TEXT ·Value·call(SB), NOSPLIT, $0
  CallImport
  RET

TEXT ·Value·invoke(SB), NOSPLIT, $0
  CallImport
  RET

TEXT ·Value·new(SB), NOSPLIT, $0
  CallImport
  RET

TEXT ·Value·Float(SB), NOSPLIT, $0
  CallImport
  RET

TEXT ·Value·Int(SB), NOSPLIT, $0
  CallImport
  RET

TEXT ·Value·Bool(SB), NOSPLIT, $0
  CallImport
  RET

TEXT ·Value·Length(SB), NOSPLIT, $0
  CallImport
  RET

TEXT ·Value·prepareString(SB), NOSPLIT, $0
  CallImport
  RET

TEXT ·Value·loadString(SB), NOSPLIT, $0
  CallImport
  RET
