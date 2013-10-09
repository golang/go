// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This .cc file will be automatically compiled by the go tool and
// included in the package.

#include <string>
#include "callback.h"

std::string Caller::call() {
	if (callback_ != 0)
		return callback_->run();
	return "";
}
